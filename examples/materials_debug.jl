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

elseif test_mode == "multi_volume"
    # Multiple volumetric materials together
    milk_medium = Hikari.Milk(scale=0.1)
    milk_glass = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.5);
        inside=milk_medium, outside=nothing
    )

    smoke_medium = Hikari.Smoke(density=5.0, albedo=0.95, g=0.3)
    smoke_vol = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.0);
        inside=smoke_medium, outside=nothing
    )

    coffee_medium = Hikari.Coffee(scale=0.5)
    coffee_glass = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(0.95, 0.9, 0.85), index=1.5);
        inside=coffee_medium, outside=nothing
    )

    [glass, milk_glass, smoke_vol, coffee_glass]

elseif test_mode == "all_volume"
    # All volumetric materials including cloud
    milk_medium = Hikari.Milk(scale=0.1)
    milk_glass = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.5);
        inside=milk_medium, outside=nothing
    )

    smoke_medium = Hikari.Smoke(density=5.0, albedo=0.95, g=0.3)
    smoke_vol = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.0);
        inside=smoke_medium, outside=nothing
    )

    coffee_medium = Hikari.Coffee(scale=0.5)
    coffee_glass = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(0.95, 0.9, 0.85), index=1.5);
        inside=coffee_medium, outside=nothing
    )

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

    [glass, milk_glass, smoke_vol, coffee_glass, cloud_vol]

elseif test_mode == "mixed"
    # Mix of volumetrics, metals, coated
    milk_medium = Hikari.Milk(scale=0.1)
    milk_glass = Hikari.MediumInterface(
        Hikari.Dielectric(Kt=(1, 1, 1), index=1.5);
        inside=milk_medium, outside=nothing
    )

    silver = Hikari.Silver(roughness=0.02)

    coated_gold = Hikari.CoatedConductor(
        interface_roughness=0.05,
        conductor_eta=(0.143, 0.374, 1.442),
        conductor_k=(3.983, 2.385, 1.603),
        conductor_roughness=0.02
    )

    emissive = Hikari.Emissive(Le=(4, 4, 4))

    [glass, milk_glass, silver, coated_gold, emissive]

elseif test_mode == "full20"
    # 20 materials like the actual scene
    function make_perlin_texture(resolution::Int; scale=4.0, bias=0.5, contrast=1.0)
        tex = Matrix{Float32}(undef, resolution, resolution)
        for j in 1:resolution, i in 1:resolution
            u, v = (i - 0.5) / resolution, (j - 0.5) / resolution
            n = Hikari.fbm3d(u * scale, v * scale, 0.0; octaves=4, persistence=0.5)
            tex[i, j] = Float32(clamp(bias + contrast * n, 0, 1))
        end
        tex
    end

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

    # Glass materials
    glass = Hikari.Dielectric(Kt=(1, 1, 1), index=1.5)
    thin_glass = Hikari.ThinDielectric(eta=1.5)
    glass_tint_tex = make_perlin_rgb_texture(64; scale=3.0, base_color=(0.95, 0.98, 1.0), variation=0.08)
    textured_glass = Hikari.Dielectric(Kt=Hikari.Texture(glass_tint_tex), index=1.5)

    # Volumetrics
    milk_medium = Hikari.Milk(scale=0.1)
    milk_glass = Hikari.MediumInterface(Hikari.Dielectric(Kt=(1, 1, 1), index=1.5); inside=milk_medium, outside=nothing)

    smoke_medium = Hikari.Smoke(density=5.0, albedo=0.95, g=0.3)
    smoke_vol = Hikari.MediumInterface(Hikari.Dielectric(Kt=(1, 1, 1), index=1.0); inside=smoke_medium, outside=nothing)

    coffee_medium = Hikari.Coffee(scale=0.5)
    coffee_glass = Hikari.MediumInterface(Hikari.Dielectric(Kt=(0.95, 0.9, 0.85), index=1.5); inside=coffee_medium, outside=nothing)

    cloud_origin = Vec3f(-0.25, -4.75, 0.0)
    cube_size = 0.5f0
    cloud_density = Hikari.generate_cloud_density(64; scale=2.5, threshold=0.15, worley_weight=0.2, edge_sharpness=4.0, density_scale=4.5)
    cloud_grid = Hikari.GridMedium(cloud_density; σ_a=Hikari.RGBSpectrum(0.5f0), σ_s=Hikari.RGBSpectrum(15.0f0), g=0.0f0,
        bounds=Hikari.Bounds3(cloud_origin, cloud_origin + Vec3f(cube_size)), majorant_res=Vec3i(16))
    cloud_vol = Hikari.MediumInterface(Hikari.Dielectric(Kt=(1, 1, 1), index=1.0); inside=cloud_grid, outside=nothing)

    # Metals
    gold_roughness_tex = make_perlin_texture(64; scale=6.0, bias=0.03, contrast=0.08)
    textured_gold = Hikari.Conductor(eta=(0.143f0, 0.374f0, 1.442f0), k=(3.983f0, 2.385f0, 1.603f0), roughness=Hikari.Texture(gold_roughness_tex))
    silver = Hikari.Silver(roughness=0.02)
    copper = Hikari.Copper(roughness=0.08)
    mirror = Hikari.Mirror(Kr=(0.95, 0.95, 0.95))

    # Coated
    coated_gold = Hikari.CoatedConductor(interface_roughness=0.05, conductor_eta=(0.143, 0.374, 1.442), conductor_k=(3.983, 2.385, 1.603), conductor_roughness=0.02)
    car_paint = Hikari.CoatedConductor(interface_roughness=0.08, reflectance=(0.85, 0.1, 0.1), conductor_roughness=0.01)
    coated_blue = Hikari.CoatedDiffuse(reflectance=(0.1, 0.2, 0.7), roughness=0.05)
    plastic_white = Hikari.Plastic(Kd=(0.9, 0.9, 0.9), Ks=(0.4, 0.4, 0.4), roughness=0.15)

    # Emissive
    emissive_white = Hikari.Emissive(Le=(4, 4, 4))
    emissive_warm = Hikari.Emissive(Le=(2.0, 1.2, 0.5))
    emissive_pattern_tex = make_perlin_rgb_texture(64; scale=5.0, base_color=(1.5, 0.3, 1.2), variation=0.8)
    textured_emissive = Hikari.Emissive(Le=Hikari.Texture(emissive_pattern_tex))

    # Simple
    diffuse_gray = Hikari.Diffuse(Kd=(0.6, 0.6, 0.6))
    paper = Hikari.DiffuseTransmission(reflectance=(0.85, 0.85, 0.85), transmittance=(0.4, 0.4, 0.4))

    [glass, textured_glass, milk_glass, smoke_vol,
     emissive_white, cloud_vol, coffee_glass, thin_glass,
     textured_gold, silver, copper, mirror,
     coated_gold, car_paint, coated_blue, plastic_white,
     emissive_warm, paper, diffuse_gray, textured_emissive]

else
    error("Unknown test mode: $test_mode")
end

println("Testing with $(length(test_materials)) materials")

# Floor
floor_material = Hikari.Diffuse(Kd=(0.7, 0.7, 0.7))
floor_mesh = Rect3f(Vec3f(-10, -10, -0.001), Vec3f(20, 20, 0.001))
mesh!(ax, floor_mesh; material=floor_material)

# Place spheres
ncols = min(length(test_materials), 4)
for (i, mat) in enumerate(test_materials)
    col = ((i - 1) % ncols) + 1
    row = ((i - 1) ÷ ncols) + 1
    x = (col - (ncols + 1) / 2) * spacing
    y = (row - 1) * spacing - 4.5
    pos = Point3f(x, y, sphere_radius)
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
