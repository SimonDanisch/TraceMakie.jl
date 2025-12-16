using GeometryBasics, Hikari
using Colors, FileIO
using Hikari
using TraceMakie
using GLMakie
using ImageShow
using Random
begin
    radiance = 50
    lights = [
        PointLight(RGBf(radiance, radiance, radiance), Vec3f(10)),
        PointLight(RGBf(10, 10, 10), Vec3f(-0.3, -5.5, 0.9)),
    ]
    fig = Figure(; size=(1024, 1024))
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(lights=lights,))

    emissive = Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.4f0, 0.6f0, 0.2f0)),
        Hikari.ConstantTexture(1.0f0),
    )
    diffuse = Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
        Hikari.ConstantTexture(0.0f0),
    )
    glass = Hikari.GlassMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
        Hikari.ConstantTexture(0.0f0),
        Hikari.ConstantTexture(0.0f0),
        Hikari.ConstantTexture(1.5f0),
        true,
    )
    plastic = Hikari.PlasticMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
        Hikari.ConstantTexture(0.010408001f0),
        true,
    )
    chrome = Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)))
    dielectric = Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0, 0f0, 0f0)))
    gold = Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0, 1.0f0, 0.0f0)))

    materials = shuffle!([glass chrome;
        gold dielectric;
        emissive plastic])

    floor = load(Makie.assetpath("matball_floor.obj"))
    mesh!(ax, floor.mesh; color=:white)

    palette = reshape(Makie.DEFAULT_PALETTES.color[][1:6], size(materials))

    for i in CartesianIndices(materials)
        x, y = Tuple(i)
        mat = materials[i]
        v = Vec3f(((x, y) .- (0.3 .* size(materials)) .- 0.3)..., 0)
        offset = 0.9 .* (v .- Vec3f(0, 5, 0))
        mesh!(ax.scene, Sphere(Point3f(offset) .+ Point3f(0, 0, 0.3), 0.3f0), material=mat)
        # translate!(mplot, 0.9 .* (v .- Vec3f(0, 3, 0)))
    end
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = Vec3f(-0.3, -6.5, 0.9)
    cam.lookat[] = Vec3f(0.5, 0, -0.5)
    cam.upvector[] = Vec3f(0, 0, 1)
    cam.fov[] = 35
    TraceMakie.render_whitted(ax.scene; samples_per_pixel=8)
    # @time colorbuffer(screen)
end

# begin
#     @time render_whitted(ax.scene)
# end
begin
    scene, cam, film = TraceMakie.convert_scene(ax.scene)
    Hikari.clear!(film)
    integrator = Hikari.WhittedIntegrator(cam[], Hikari.UniformSampler(8), 8)
    @time integrator(scene, film, cam[])
    film.framebuffer
end
