using GeometryBasics, Hikari
using Colors, FileIO
using TraceMakie
using GLMakie
using ImageShow
using pocl_jll, OpenCL

# ============================================================================
# Material Gallery Scene
# ============================================================================

begin
    # Setup lighting
    radiance = 50
    lights = [
        PointLight(RGBf(radiance, radiance, radiance), Vec3f(10)),
        PointLight(RGBf(15, 15, 15), Vec3f(-0.3, -5.5, 1.5)),
    ]

    ax = Scene(; size=(1200, 900), lights=lights)
    cam3d!(ax)
    # ========================================================================
    # Define materials using the clean API
    # ========================================================================

    # Row 1: Basic materials
    diffuse_red = Hikari.Diffuse(Kd=(0.8, 0.2, 0.2))
    diffuse_green = Hikari.Diffuse(Kd=(0.2, 0.8, 0.3), σ=30)  # Oren-Nayar roughness
    mirror = Hikari.Mirror(Kr=(0.95, 0.95, 0.95))
    glass = Hikari.Dielectric(Kt=(1, 1, 1), index=1.5)

    # Row 2: Metals using preset constructors
    gold = Hikari.Gold(roughness=0.05)
    silver = Hikari.Silver(roughness=0.02)
    copper = Hikari.Copper(roughness=0.1)
    aluminum = Hikari.Aluminum(roughness=0.15)

    # Row 3: Plastic and coated diffuse materials
    plastic_blue = Hikari.Plastic(Kd=(0.1, 0.2, 0.8), Ks=(0.5, 0.5, 0.5), roughness=0.05)
    plastic_white = Hikari.Plastic(Kd=(0.9, 0.9, 0.9), Ks=(0.3, 0.3, 0.3), roughness=0.2)
    coated_red = Hikari.CoatedDiffuse(reflectance=(0.8, 0.2, 0.2), roughness=0.1)
    coated_green = Hikari.CoatedDiffuse(reflectance=(0.2, 0.7, 0.3), roughness=0.0)

    # Row 4: New materials - ThinDielectric, DiffuseTransmission, CoatedConductor
    thin_glass = Hikari.ThinDielectric(eta=1.5)  # Window glass
    paper = Hikari.DiffuseTransmission(  # Paper/cloth
        reflectance=(0.8, 0.8, 0.8),
        transmittance=(0.5, 0.5, 0.5)
    )
    coated_gold = Hikari.CoatedConductor(  # Lacquered gold
        interface_roughness=0.05,
        conductor_eta=(0.143, 0.374, 1.442),  # Gold
        conductor_k=(3.983, 2.385, 1.603),
        conductor_roughness=0.02
    )
    car_paint = Hikari.CoatedConductor(  # Red metallic car paint
        interface_roughness=0.1,
        reflectance=(0.9, 0.1, 0.1),  # Red metallic
        conductor_roughness=0.01
    )

    # Arrange in grid (4 rows x 4 columns)
    materials = [
        diffuse_red    diffuse_green  mirror         glass;
        gold           silver         copper         aluminum;
        plastic_blue   plastic_white  coated_red     coated_green;
        thin_glass     paper          coated_gold    car_paint
    ]

    labels = [
        "Diffuse"      "Diffuse+σ"    "Mirror"       "Glass";
        "Gold"         "Silver"       "Copper"       "Aluminum";
        "Plastic"      "Plastic"      "CoatedDiff"   "CoatedDiff";
        "ThinGlass"    "Paper"        "CoatedGold"   "CarPaint"
    ]

    # Load floor
    floor = load(Makie.assetpath("matball_floor.obj"))
    mesh!(ax, floor.mesh; color=:white)

    # Place spheres in grid
    sphere_radius = 0.25f0
    spacing = 0.7f0

    nrows, ncols = size(materials)
    for i in CartesianIndices(materials)
        row, col = Tuple(i)
        mat = materials[i]

        # Center the grid
        x = (col - (ncols + 1) / 2) * spacing
        y = (row - (nrows + 1) / 2) * spacing - 4.5  # Offset towards camera

        pos = Point3f(x, y, sphere_radius)
        mesh!(ax, Sphere(pos, sphere_radius), material=mat)
    end

    # Camera setup
    cam = cameracontrols(ax)
    cam.eyeposition[] = Vec3f(0, -7, 2)
    cam.lookat[] = Vec3f(0, -3, 0)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 40
end
# Render with VolPath integrator
TraceMakie.activate!(backend=CLArray)
integrator = Hikari.VolPath(samples=1, regularize=true, max_depth=8, material_coherence=:none)
img = @time "per material" colorbuffer(ax; backend=TraceMakie, integrator=integrator)
screen = Makie.getscreen(ax)
img = @time "per material" colorbuffer(screen)

img2 = @time "normal" colorbuffer(ax; backend=TraceMakie, integrator=Hikari.VolPath(samples=32, regularize=true, max_depth=8))
