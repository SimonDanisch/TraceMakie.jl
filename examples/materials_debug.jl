using GeometryBasics, Hikari
using Colors, FileIO
using TraceMakie
using Makie
using pocl_jll, OpenCL

# Minimal test to find OpenCL crash
sphere_radius = 0.25f0
spacing = 0.7f0

lights = [
    PointLight(RGBf(60, 60, 60), Vec3f(8, 8, 10)),
]

ax = Scene(; size=(400, 300), lights=lights, ambient=RGBf(0.02, 0.02, 0.025))
cam3d!(ax)

# Base materials
glass = Hikari.Dielectric(Kt=(1, 1, 1), index=1.5)
diffuse_gray = Hikari.Diffuse(Kd=(0.6, 0.6, 0.6))

# Test mode from command line
test_mode = length(ARGS) > 0 ? ARGS[1] : "base"
println("Test mode: $test_mode")

test_materials = if test_mode == "base"
    [glass, diffuse_gray]

elseif test_mode == "milk"
    milk_medium = Hikari.Milk(scale=0.1)
    milk_glass = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.5);
        inside=milk_medium, outside=nothing
    )
    [glass, milk_glass]

elseif test_mode == "smoke"
    smoke_medium = Hikari.Smoke(density=5.0, albedo=0.95, g=0.3)
    smoke_vol = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.0);
        inside=smoke_medium, outside=nothing
    )
    [glass, smoke_vol]

elseif test_mode == "cloud"
    # GridMedium - the most complex one
    cloud_origin = Vec3f(-0.25, -4.75, 0.0)
    cube_size = 0.5f0
    cloud_density = Hikari.generate_cloud_density(64;
        scale=2.5, threshold=0.15, worley_weight=0.2,
        edge_sharpness=4.0, density_scale=4.5
    )
    cloud_grid = Hikari.GridMedium(
        cloud_density;
        σ_a = Hikari.RGBSpectrum(0.5f0),
        σ_s = Hikari.RGBSpectrum(15.0f0),
        g = 0.0f0,
        bounds=Hikari.Bounds3(cloud_origin, cloud_origin + Vec3f(cube_size)),
        majorant_res=Vec3i(16)
    )
    cloud_vol = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.0);
        inside=cloud_grid, outside=nothing
    )
    [glass, cloud_vol]

elseif test_mode == "textured"
    # Textured material - might cause issues
    function make_perlin_texture(resolution::Int; scale=4.0, bias=0.5, contrast=1.0)
        tex = Matrix{Float32}(undef, resolution, resolution)
        for j in 1:resolution, i in 1:resolution
            u, v = (i - 0.5) / resolution, (j - 0.5) / resolution
            n = Hikari.fbm3d(u * scale, v * scale, 0.0; octaves=4, persistence=0.5)
            tex[i, j] = Float32(clamp(bias + contrast * n, 0, 1))
        end
        tex
    end
    gold_roughness_tex = make_perlin_texture(64; scale=6.0, bias=0.03, contrast=0.08)
    textured_gold = Hikari.Conductor(
        eta = (0.143f0, 0.374f0, 1.442f0),
        k = (3.983f0, 2.385f0, 1.603f0),
        roughness = Hikari.Texture(gold_roughness_tex)
    )
    [glass, textured_gold]

elseif test_mode == "emissive_tex"
    # Textured emissive
    function make_perlin_rgb_texture(resolution::Int; scale=4.0, base_color=(1.0, 1.0, 1.0), variation=0.3)
        tex = Matrix{Hikari.RGBSpectrum}(undef, resolution, resolution)
        for j in 1:resolution, i in 1:resolution
            u, v = (i - 0.5) / resolution, (j - 0.5) / resolution
            n = Hikari.fbm3d(u * scale, v * scale, 0.0; octaves=4)
            n2 = Hikari.fbm3d(u * scale + 5.3, v * scale - 2.1, 0.0; octaves=3)
            r = Float32(clamp(base_color[1] + variation * n, 0, 1))
            g = Float32(clamp(base_color[2] + variation * n2, 0, 1))
            b = Float32(clamp(base_color[3] + variation * (n + n2) * 0.5, 0, 1))
            tex[i, j] = Hikari.RGBSpectrum(r, g, b, 1f0)
        end
        tex
    end
    emissive_pattern_tex = make_perlin_rgb_texture(64; scale=5.0, base_color=(1.5, 0.3, 1.2), variation=0.8)
    textured_emissive = Hikari.Emissive(Le=Hikari.Texture(emissive_pattern_tex))
    [glass, textured_emissive]

elseif test_mode == "coated"
    coated_gold = Hikari.CoatedConductor(
        interface_roughness=0.05,
        conductor_eta=(0.143, 0.374, 1.442),
        conductor_k=(3.983, 2.385, 1.603),
        conductor_roughness=0.02
    )
    [glass, coated_gold]

elseif test_mode == "all_simple"
    # All simple materials without volumes
    silver = Hikari.Silver(roughness=0.02)
    copper = Hikari.Copper(roughness=0.08)
    mirror = Hikari.Mirror(Kr=(0.95, 0.95, 0.95))
    emissive = Hikari.Emissive(Le=(4, 4, 4))
    [glass, diffuse_gray, silver, copper, mirror, emissive]

elseif test_mode == "full"
    # Try the full material set
    include("materials_full_test.jl")

else
    error("Unknown test mode: $test_mode")
end

println("Testing with $(length(test_materials)) materials")

# Floor
floor_material = Hikari.Diffuse(Kd=(0.7, 0.7, 0.7))
floor_mesh = Rect3f(Vec3f(-10, -10, -0.001), Vec3f(20, 20, 0.001))
mesh!(ax, floor_mesh; material=floor_material)

# Place spheres
for (i, mat) in enumerate(test_materials)
    x = (i - (length(test_materials) + 1) / 2) * spacing
    pos = Point3f(x, -4.5, sphere_radius)
    mesh!(ax, Sphere(pos, sphere_radius), material=mat)
end

# Camera
cam = cameracontrols(ax)
cam.eyeposition[] = Vec3f(0, -7.5, 2.5)
cam.lookat[] = Vec3f(0, -4.5, 0)
cam.upvector[] = Vec3f(0, 0, 1)
cam.fov[] = 42
update_cam!(ax, cam)

# Render with OpenCL
println("Activating TraceMakie with CLArray backend...")
TraceMakie.activate!(backend=CLArray,
    exposure=0.6f0,
    tonemap=:aces,
    gamma=2.2f0,
    sensor=Hikari.FilmSensor(iso=50, exposure_time=1.0, white_balance=0)
)
integrator = Hikari.VolPath(samples=4, max_depth=10)
println("Starting render...")
img = @time colorbuffer(ax; backend=TraceMakie, integrator=integrator)
println("Render complete!")
save(joinpath(@__DIR__, "debug_$(test_mode).png"), img)
println("Saved to debug_$(test_mode).png")
