# Bunny Cloud Scene - NanoVDB Volumetric Path Tracing Example
# Uses actual NanoVDB volumetric data from pbrt-v4-scenes for spatially-varying density
#
# This parses the NanoVDB file format directly in Julia and renders with GridMedium + VolPath
using TraceMakie, Makie, Hikari, GeometryBasics, Raycore
using FileIO
using Zlib_jll
using pocl_jll, OpenCL
# =============================================================================
# NanoVDB Parser
# =============================================================================

# Zlib stream structure for decompression
mutable struct ZStream
    next_in::Ptr{UInt8}
    avail_in::Cuint
    total_in::Culong
    next_out::Ptr{UInt8}
    avail_out::Cuint
    total_out::Culong
    msg::Ptr{Cchar}
    state::Ptr{Cvoid}
    zalloc::Ptr{Cvoid}
    zfree::Ptr{Cvoid}
    opaque::Ptr{Cvoid}
    data_type::Cint
    adler::Culong
    reserved::Culong
end

ZStream() = ZStream(
    C_NULL, 0, 0,
    C_NULL, 0, 0,
    C_NULL, C_NULL,
    C_NULL, C_NULL, C_NULL,
    0, 0, 0
)

"""
    decompress_zlib(compressed_data::Vector{UInt8}, output_size::Int) -> Vector{UInt8}

Decompress zlib-compressed data.
"""
function decompress_zlib(compressed_data::Vector{UInt8}, output_size::Int)
    output_buffer = Vector{UInt8}(undef, output_size)

    z = Ref(ZStream())
    z[].next_in = pointer(compressed_data)
    z[].avail_in = length(compressed_data)
    z[].next_out = pointer(output_buffer)
    z[].avail_out = output_size

    ret = ccall((:inflateInit_, Zlib_jll.libz), Cint,
                (Ref{ZStream}, Cstring, Cint), z, "1.2.11", sizeof(ZStream))
    ret != 0 && error("inflateInit failed: $ret")

    ret = ccall((:inflate, Zlib_jll.libz), Cint, (Ref{ZStream}, Cint), z, 4)  # Z_FINISH=4
    decompressed_size = z[].total_out
    ccall((:inflateEnd, Zlib_jll.libz), Cint, (Ref{ZStream},), z)

    return output_buffer[1:decompressed_size]
end

"""
    parse_nanovdb(filepath::String) -> (volume::Array{Float32,3}, world_bounds::Tuple)

Parse a NanoVDB file and return the density volume and world bounds.
"""
function parse_nanovdb(filepath::String)
    nvdb_data = read(filepath)
    println("NanoVDB file size: $(length(nvdb_data)) bytes")

    # Find zlib header (0x78 0x9c for default compression)
    compressed_start = 0
    for i in 1:min(500, length(nvdb_data)-1)
        if nvdb_data[i] == 0x78 && nvdb_data[i+1] in (0x01, 0x5e, 0x9c, 0xda)
            compressed_start = i
            break
        end
    end
    compressed_start == 0 && error("Could not find zlib header")
    println("Found zlib header at byte $compressed_start")

    # Decompress grid data
    compressed_data = nvdb_data[compressed_start:end]
    println("Decompressing $(length(compressed_data)) bytes...")
    grid_data = decompress_zlib(compressed_data, 200_000_000)
    println("Decompressed to $(length(grid_data)) bytes")

    # Read world bounds from GridData header (offset 561-608 for world bbox)
    world_bbox = reinterpret(Float64, grid_data[561:608])
    world_min = (Float32(world_bbox[1]), Float32(world_bbox[2]), Float32(world_bbox[3]))
    world_max = (Float32(world_bbox[4]), Float32(world_bbox[5]), Float32(world_bbox[6]))
    println("World bounds: $world_min to $world_max")

    # Read leaf count from TreeData (offset 673 + 32 for node counts)
    # TreeData: 4 uint64 offsets (32 bytes) + 4 uint32 counts (16 bytes)
    tree_offset = 673
    node_counts = reinterpret(UInt32, grid_data[tree_offset+32:tree_offset+47])
    leaf_count = Int(node_counts[1])  # First count is leaf nodes
    println("Found $leaf_count leaf nodes")

    # Each leaf: 12 bytes coords + 4 bytes bbox/flags + 64 bytes mask + 2048 bytes values + 16 bytes stats = 2144 bytes
    leaf_size = 2144

    # Leaf data is at the end of the grid - calculate start from total size
    total_leaf_bytes = leaf_count * leaf_size
    leaf_data_offset = length(grid_data) - total_leaf_bytes + 1
    println("Leaf data starts at offset: $leaf_data_offset")

    # Read all leaf coordinates to get volume bounds
    all_coords = Vector{NTuple{3,Int32}}(undef, leaf_count)
    for i in 0:leaf_count-1
        offset = leaf_data_offset + i * leaf_size
        c = reinterpret(Int32, grid_data[offset:offset+11])
        all_coords[i+1] = (c[1], c[2], c[3])
    end

    min_x, max_x = extrema(c[1] for c in all_coords)
    min_y, max_y = extrema(c[2] for c in all_coords)
    min_z, max_z = extrema(c[3] for c in all_coords)
    println("Leaf coord bounds: X=$min_x:$max_x, Y=$min_y:$max_y, Z=$min_z:$max_z")

    # Create volume array
    vol_size_x = max_x - min_x + 8
    vol_size_y = max_y - min_y + 8
    vol_size_z = max_z - min_z + 8
    println("Creating volume: $vol_size_x x $vol_size_y x $vol_size_z")

    volume = zeros(Float32, vol_size_x, vol_size_y, vol_size_z)

    # Fill volume from leaves
    for leaf_idx in 0:leaf_count-1
        offset = leaf_data_offset + leaf_idx * leaf_size
        cx, cy, cz = reinterpret(Int32, grid_data[offset:offset+11])

        # Read 512 float values (8x8x8 block)
        vals_offset = offset + 80
        vals = reinterpret(Float32, grid_data[vals_offset:vals_offset+2047])

        # Map to volume indices
        base_x = cx - min_x + 1
        base_y = cy - min_y + 1
        base_z = cz - min_z + 1

        # Fill 8x8x8 block (NanoVDB uses x-major ordering)
        idx = 1
        for lx in 0:7
            for ly in 0:7
                for lz in 0:7
                    vx = base_x + lx
                    vy = base_y + ly
                    vz = base_z + lz
                    if 1 <= vx <= vol_size_x && 1 <= vy <= vol_size_y && 1 <= vz <= vol_size_z
                        volume[vx, vy, vz] = vals[idx]
                    end
                    idx += 1
                end
            end
        end
    end

    nonzero = count(x -> x > 0, volume)
    println("Volume filled: $nonzero non-zero voxels")

    return volume, world_min, world_max
end

"""
    downsample_volume(volume::Array{Float32,3}, factor::Int) -> Array{Float32,3}

Downsample volume by averaging blocks.
"""
function downsample_volume(volume::Array{Float32,3}, factor::Int)
    sx, sy, sz = size(volume)
    ds_size = (sx ÷ factor, sy ÷ factor, sz ÷ factor)
    result = zeros(Float32, ds_size...)

    for ix in 1:ds_size[1]
        for iy in 1:ds_size[2]
            for iz in 1:ds_size[3]
                total = 0f0
                count = 0
                for dx in 0:factor-1, dy in 0:factor-1, dz in 0:factor-1
                    x = (ix-1)*factor + dx + 1
                    y = (iy-1)*factor + dy + 1
                    z = (iz-1)*factor + dz + 1
                    if x <= sx && y <= sy && z <= sz
                        total += volume[x, y, z]
                        count += 1
                    end
                end
                result[ix, iy, iz] = total / count
            end
        end
    end

    return result
end

# =============================================================================
# Scene Creation - Matching pbrt-v4 bunny-cloud.pbrt exactly
# =============================================================================

# pbrt scene reference:
# LookAt 00 120 50       7 0 17   0 0 1
# Camera "perspective" "float fov" 25
# Film "rgb" "string sensor" "nikon_d850" "float whitebalance" 5000 "float iso" 90
# Integrator "volpath" "integer maxdepth" 50
#
# LightSource "infinite" "string filename" "textures/sky.exr" "float scale" 4
#   (rotated 10° around X axis)
#
# MakeNamedMedium "foo" "string type" "nanovdb" "string filename" "bunny_cloud.nvdb"
#   "spectrum sigma_s" [200 10 900 10] "spectrum sigma_a" [200 .5 900 .5]
#   (rotated 180° around Z, then 90° around X)
#
# MediumInterface "foo" "" + Material "interface" + Shape "sphere" "float radius" 45
#
# Ground: Material "coateddiffuse" "rgb reflectance" [ .4 .45 .35 ] "float roughness" 0
#         Shape "disk" "float radius" 1000 (translated y=-50)

"""
    create_nanovdb_bunny_scene(nvdb_path::String; resolution=(800,600), downsample=4)

Create a bunny cloud scene from NanoVDB file, matching pbrt-v4 bunny-cloud.pbrt exactly.
"""
function create_nanovdb_bunny_scene(nvdb_path::String;
    resolution=(800, 600),
    downsample=4,
    # pbrt-v4: "spectrum sigma_s" [200 10 900 10]
    # This is piecewise linear spectrum: [λ₁ value₁ λ₂ value₂]
    # For this scene: [200nm→10, 900nm→10] is a flat spectrum (constant value 10)
    # We use scalar since RGBSpectrum gets uplifted at runtime (equivalent to flat spectrum)
    sigma_s=10.0f0,
    # pbrt-v4: "spectrum sigma_a" [200 .5 900 .5]
    # Flat spectrum: constant 0.5 across wavelengths
    sigma_a=0.5f0,
    # Phase function asymmetry parameter (g=0 means isotropic scattering)
    g=0.0f0
)
    # Parse NanoVDB file
    volume, world_min, world_max = parse_nanovdb(nvdb_path)

    # Downsample for practical rendering
    volume_ds = downsample_volume(volume, downsample)
    println("Downsampled to: $(size(volume_ds))")

    # NanoVDB stores Y-up bunny, so swap Y<->Z
    volume_reoriented = permutedims(volume_ds, (1, 3, 2))
    # Flip X to match the 180° Z rotation (bunny facing correct direction)
    volume_reoriented = reverse(volume_reoriented, dims=1)

    # Compute world bounds after reorientation
    # Original bounds are in Y-up space, convert to Z-up
    new_world_min = Vec3f(-world_max[1], world_min[3], world_min[2])
    new_world_max = Vec3f(-world_min[1], world_max[3], world_max[2])

    # Ensure min < max
    actual_min = Vec3f(min(new_world_min[1], new_world_max[1]),
                       min(new_world_min[2], new_world_max[2]),
                       min(new_world_min[3], new_world_max[3]))
    actual_max = Vec3f(max(new_world_min[1], new_world_max[1]),
                       max(new_world_min[2], new_world_max[2]),
                       max(new_world_min[3], new_world_max[3]))

    # Normalize density
    volume_norm = volume_reoriented ./ maximum(volume_reoriented)

    # Create GridMedium with pbrt-matching sigma values
    # RGBSpectrum(scalar) creates a flat RGB spectrum that gets uplifted to hero wavelengths
    # at runtime via uplift_rgb_unbounded(). This matches pbrt's behavior for flat spectra
    # where DenselySampledSpectrum.Sample(lambda) returns the same value for all wavelengths.
    bounds = Raycore.Bounds3(actual_min, actual_max)
    grid_medium = Hikari.GridMedium(
        volume_norm;
        σ_a = Hikari.RGBSpectrum(sigma_a),
        σ_s = Hikari.RGBSpectrum(sigma_s),
        g = g,
        bounds = bounds
    )
    @show bounds
    # Create scene (no default lights - we'll add environment light)
    s = Scene(size=resolution; lights=Makie.AbstractLight[])
    cam3d!(s)

    # Camera setup - matching pbrt exactly:
    # LookAt 00 120 50       7 0 17   0 0 1
    # Camera at (0, 120, 50), looking at (7, 0, 17), up is (0, 0, 1)
    # Camera "perspective" "float fov" 25
    cam_pos = Vec3f(0, 120, 50)
    look_at = Vec3f(7, 0, 17)
    update_cam!(s, cam_pos, look_at, Vec3f(0, 0, 1))

    # Set FOV to match pbrt (25 degrees)
    s.camera_controls.fov[] = 25.0

    # Transparent boundary material for medium interface (pbrt's "interface" material)
    transparent = Hikari.GlassMaterial(
        Kr = Hikari.RGBSpectrum(0f0),
        Kt = Hikari.RGBSpectrum(1f0),
        index = 1.0f0
    )

    # Create sphere geometry for medium boundary (pbrt uses sphere radius 45)
    # The sphere is centered at origin in pbrt (0, 0, 0)
    # With radius 45, bottom of sphere is at Z = -45, close to ground at Z = -50
    sphere_mesh = GeometryBasics.normal_mesh(GeometryBasics.Sphere(Point3f(0, 0, 0), 45f0))

    # Volume sphere with medium interface
    volume_material = Hikari.MediumInterface(transparent; inside=grid_medium, outside=nothing)
    mesh!(s, sphere_mesh; material=volume_material)

    # Ground plane - pbrt uses:
    # Translate 0 -50 0
    # Material "coateddiffuse" "rgb reflectance" [ .4 .45 .35 ] "float roughness" 0
    # Shape "disk" "float radius" 1000
    #
    # pbrt disk default orientation: lies in XY plane with normal pointing +Z
    # After Translate 0 -50 0: disk center moves to (0, -50, 0)
    # But the disk is still HORIZONTAL (normal +Z) - it's a ground plane!
    # Since Z is up in this scene, the ground should be at Z = -50 (UNDER the bunny)
    # Note: pbrt's "Translate 0 -50 0" in Y doesn't change Z position of disk surface

    # Create a large flat box for the ground at Z=0 (horizontal, Z is up)
    # pbrt disk is at Z=0 (Translate 0 -50 0 only moves in Y, not Z)
    # The sphere (radius 45 at origin) intersects the ground - this is intentional!
    # The bunny cloud density is mostly above Z=0, so ground cuts through lower sphere
    ground_size = 1000f0
    ground_z = 0f0  # Z position (ground plane at Z=0, sphere intersects it)
    ground_geo = Rect3f(Vec3f(-ground_size, -ground_size, ground_z - 0.1f0),
                        Vec3f(2*ground_size, 2*ground_size, 0.2f0))

    # CoatedDiffuseMaterial - exact pbrt-v4 port with LayeredBxDF
    # "coateddiffuse" "rgb reflectance" [ .4 .45 .35 ] "float roughness" 0
    ground_material = Hikari.CoatedDiffuseMaterial(
        reflectance = (0.4f0, 0.45f0, 0.35f0),
        roughness = 0f0,  # Smooth coating (specular top layer)
        eta = 1.5f0,      # Default coating IOR (typical for dielectric)
        thickness = 0.01f0
    )
    mesh!(s, ground_geo; material=ground_material)

    # Environment lighting - pbrt uses:
    # Rotate 10 1 0 0  (10° around X axis)
    # LightSource "infinite" "string filename" "textures/sky.exr" "float scale" 4
    sky_path = joinpath(dirname(nvdb_path), "textures", "sky.exr")

    # Load sky.exr and create Makie.EnvironmentLight
    # Makie.EnvironmentLight expects (intensity, image; rotation_angle, rotation_axis)
    sky_image = FileIO.load(sky_path)
    # Convert to Matrix{RGBf} and scale by 4.0 (pbrt's scale factor)
    sky_matrix = Matrix{RGBf}(map(c -> RGBf(c.r, c.g, c.b), sky_image))
    # Apply 10° rotation around X axis to match pbrt's "Rotate 10 1 0 0"
    env_light = Makie.EnvironmentLight(4.0f0, sky_matrix;
        rotation_angle=10f0, rotation_axis=Vec3f(1, 0, 0))
    push_light!(s, env_light)
    println("Loaded environment light from: $sky_path")
    return s
end


"""
    render_nanovdb_bunny(nvdb_path; resolution=(800,600), spp=32, max_depth=50, kwargs...)

Render the NanoVDB bunny cloud scene, matching pbrt-v4 bunny-cloud.pbrt settings.

# pbrt-v4 reference settings:
# - Film "rgb" "string sensor" "nikon_d850" "float whitebalance" 5000 "float iso" 90
# - Camera "perspective" "float fov" 25
# - Integrator "volpath" "integer maxdepth" 50

# Keyword Arguments
- `resolution`: Image resolution (width, height), default (1920, 1080) matching pbrt
- `samples_per_pixel`: Number of samples per pixel, default 32
- `max_depth`: Maximum path depth, default 50 (matches pbrt)
- `exposure`: Exposure multiplier, default 1.0
- `iso`: Film sensor ISO, default 90 (matches pbrt bunny-cloud scene)
- `white_balance`: White balance color temperature in Kelvin, default 5000 (matches pbrt)
- `tonemap`: Tonemapping method (:aces, :reinhard, :filmic, etc.), default :aces
- `gamma`: Gamma correction, default 2.2
- `backend`: Compute backend (Array for CPU), default Array
"""
function render_nanovdb_bunny(nvdb_path::String;
    resolution=(1920, 1080),
    samples_per_pixel=32,
    max_depth=50,
    exposure=1.0f0,
    iso=90f0,
    white_balance=5000f0,
    tonemap=:aces,
    gamma=2.2f0,
    backend=Array,
    kwargs...
)
    # Configure VolPath integrator with pbrt-matching sensor settings
    volpath_config = (
        backend = backend,
        integrator = TraceMakie.VolPath(
            samples=samples_per_pixel,
            max_depth=max_depth
        ),
        exposure = exposure,
        tonemap = tonemap,
        gamma = gamma,
        sensor = Hikari.FilmSensor(iso=iso, white_balance=white_balance),
    )
    TraceMakie.activate!(; volpath_config...)

    # Create and render scene
    scene = create_nanovdb_bunny_scene(nvdb_path; resolution=resolution, kwargs...)
    img = colorbuffer(scene; backend=TraceMakie)

    return img, scene
end
nvdb_path = joinpath(@__DIR__, "..", "..", "..", "pbrt-v4-scenes", "bunny-cloud", "bunny_cloud.nvdb")

# scene = create_nanovdb_bunny_scene(nvdb_path);
# display(scene; backend=GLMakie)

# =============================================================================
# Example Usage - Matching pbrt-v4 bunny-cloud.pbrt
# =============================================================================

# Path to NanoVDB file

if !isfile(nvdb_path)
    error("NanoVDB file not found at: $nvdb_path\n" *
            "Please download pbrt-v4-scenes or adjust the path.")
end

println("Rendering NanoVDB bunny cloud (matching pbrt-v4 settings)...")
@time img, scene = render_nanovdb_bunny(
    nvdb_path;
    resolution=(800, 600),      # Lower res for testing (pbrt uses 1920x1080)
    samples_per_pixel=100,
    max_depth=50,               # Matches pbrt
    downsample=2,
    # pbrt-matching parameters (now defaults):
    # sigma_s=10.0, sigma_a=0.5, g=0.0
    # iso=90, , fov=25,
    white_balance=5000,
    backend=CLArray
)
img

screen = Makie.getscreen(scene)

Array(Hikari.postprocess!(screen.state.film;
    exposure=1f0,
    tonemap=nothing,
    gamma=2.2f0,
    sensor=Hikari.FilmSensor(iso=20, white_balance=4000)
))

# Save result
# output_path = joinpath(@__DIR__, "bunny_cloud_nanovdb.png")
save(output_path, img)
println("Saved to: $output_path")
1
