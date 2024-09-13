using GeometryBasics, Trace
using Colors, FileIO
using Trace
using TraceMakie
using GLMakie
using ImageShow
using Random
begin
    radiance = 1000
    lights = [
        PointLight(Vec3f(10), RGBf(radiance, radiance, radiance * 1.1)),
        PointLight(Vec3f(-0.3, -5.5, 0.9), RGBf(50, 50, 50)),
    ]
    fig = Figure(; size=(1024, 1024))
    ax = LScene(fig[1, 1]; show_axis=false, scenekw=(lights=lights,))

    emissive = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.4f0, 0.6f0, 0.2f0)),
        Trace.ConstantTexture(1.0f0),
    )
    diffuse = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
        Trace.ConstantTexture(0.0f0),
    )
    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(1.5f0),
        true,
    )
    plastic = Trace.PlasticMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
        Trace.ConstantTexture(0.010408001f0),
        true,
    )
    chrome = Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)))
    dielectric = Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0, 0f0, 0f0)))
    gold = Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0, 1.0f0, 0.0f0)))

    materials = shuffle!([glass chrome;
        gold dielectric;
        emissive plastic])

    mesh!(ax, load(Makie.assetpath("matball_floor.obj")); color=:white)

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
    # @time colorbuffer(screen)
end


# begin
#     @time render_whitted(ax.scene)
# end
begin
    scene, cam, film = TraceMakie.convert_scene(ax.scene)
    Trace.clear!(film)
    w = Trace.Whitten5(film; samples_per_pixel=8, max_depth=8)
    @time Trace.launch_trace_image!(w, cam[], scene)
    Trace.to_framebuffer!(film.framebuffer, w.pixel)
end
