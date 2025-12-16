module TraceMakie

using Makie, Hikari, Colors, LinearAlgebra, GeometryBasics, Raycore

function to_spectrum(data::Colorant)
    rgb = RGBf(data)
    return Hikari.RGBSpectrum(rgb.r, rgb.g, rgb.b)
end

function to_spectrum(data::AbstractMatrix{<:Colorant})
    colors = convert(AbstractMatrix{RGBf}, data)
    return collect(reinterpret(Hikari.RGBSpectrum, colors))
end

function extract_material(plot::Plot, tex::Union{Hikari.Texture, Nothing})
    if haskey(plot, :material) && !isnothing(to_value(plot.material))
        if to_value(plot.material) isa Hikari.Material
            return to_value(plot.material)
        end
    elseif tex isa Nothing
        error("Neither color nor material are defined for plot: $plot")
    else
        return Hikari.MatteMaterial(tex, Hikari.ConstantTexture(0.0f0))
    end
end

function extract_material(plot::Plot, color_obs::Union{Makie.Computed, Observable})
    color = to_value(color_obs)
    tex = nothing
    if color isa AbstractMatrix{<:Number}
        calc_color = to_value(plot.calculated_colors)
        tex = Hikari.Texture(to_spectrum(to_color(calc_color)))
        onany(plot, color_obs, plot.colormap, plot.colorrange) do color, cmap, crange
            tex.data = to_spectrum(to_color(calc_color))
            return
        end
    elseif color isa AbstractMatrix{<:Colorant}
        tex = Hikari.Texture(to_spectrum(color))
        onany(plot, color_obs) do color
            tex.data = to_spectrum(color)
            return
        end
    elseif color isa Colorant || color isa Union{String,Symbol}
        tex = Hikari.ConstantTexture(to_spectrum(to_color(color)))
    elseif color isa Nothing
        # ignore!
        nothing
    else
        error("Unsupported color type for RadeonProRender backend: $(typeof(color))")
    end

    return extract_material(plot, tex)
end

function to_trace_primitive(plot::Makie.Mesh)
    # Potentially per instance attributes
    mesh = plot.mesh[]
    # Convert to TriangleMesh using Raycore
    tmesh = Raycore.TriangleMesh(mesh)
    material = extract_material(plot, plot.color)
    return (tmesh, material)
end

function to_trace_primitive(plot::Makie.Surface)
    !plot.visible[] && return nothing
    x = plot[1]
    y = plot[2]
    z = plot[3]

    function grid(x, y, z, trans)
        space = to_value(get(plot, :space, :data))
        g = map(CartesianIndices(z)) do i
            p = Point3f(Makie.get_dim(x, i, 1, size(z)), Makie.get_dim(y, i, 2, size(z)), z[i])
            return Makie.apply_transform(trans, p, space)
        end
        return vec(g)
    end

    positions = lift(grid, x, y, z, Makie.transform_func_obs(plot))
    # normals = Makie.surface_normals(x[], y[], z[])
    r = Tesselation(Rect2f((0, 0), (1, 1)), size(z[]))
    # decomposing a rectangle into uv and triangles is what we need to map the z coordinates on
    # since the xyz data assumes the coordinates to have the same neighouring relations
    # like a grid
    faces = decompose(GLTriangleFace, r)
    uv = decompose_uv(r)
    # with this we can build a mesh
    mesh = normal_mesh(GeometryBasics.Mesh(vec(positions[]), faces, uv=uv))

    # Convert to TriangleMesh using Raycore
    tmesh = Raycore.TriangleMesh(mesh)
    material = extract_material(plot, plot.z)
    return Hikari.GeometricPrimitive(tmesh, material)
end

function to_trace_primitive(plot::Makie.Plot)
    return nothing
end

function to_trace_light(light::Makie.AmbientLight)
    color = light.color isa Observable ? light.color[] : light.color
    return Hikari.AmbientLight(
        to_spectrum(color),
    )
end

function to_trace_light(light::Makie.PointLight)
    return Hikari.PointLight(
        Vec3f(light.position), to_spectrum(light.color),
    )
end

function to_trace_light(light)
    return nothing
end

function to_trace_camera(scene::Makie.Scene, film)
    cc = scene.camera_controls
    return lift(scene, cc.eyeposition, cc.lookat, cc.upvector, cc.fov) do eyeposition, lookat, upvector, fov
        view = Hikari.look_at(
            Point3f(eyeposition), Point3f(lookat), Vec3f(upvector),
        )
        return Hikari.PerspectiveCamera(
            view, Hikari.Bounds2(Point2f(-1.0f0), Point2f(1.0f0)),
            0.0f0, 1.0f0, 0.0f0, 1.0f6, Float32(fov),
            film
        )
    end
    return
end

function convert_scene(scene::Makie.Scene)
        # Only set background image if it isn't set by env light, since
    # background image takes precedence
    resolution = Point2f(size(scene))
    f = Hikari.LanczosSincFilter(Point2f(1.0f0), 3.0f0)
    film = Hikari.Film(
        resolution,
        Hikari.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
    )
    primitives = Tuple[]
    for plot in scene.plots
        prim = to_trace_primitive(plot)
        !isnothing(prim) && push!(primitives, prim)
    end
    camera = to_trace_camera(scene, film)

    # Extract lights from the new compute graph structure
    lights = []
    makie_lights = Makie.get_lights(scene)
    for light in makie_lights
        l = to_trace_light(light)
        isnothing(l) || push!(lights, l)
    end

    # Add ambient light if present
    if haskey(scene.compute, :ambient_color)
        ambient_color = scene.compute[:ambient_color][]
        if ambient_color != RGBf(0, 0, 0)
            push!(lights, Hikari.AmbientLight(to_spectrum(ambient_color)))
        end
    end

    if isempty(lights)
        error("Must have at least one light")
    end
    bvh = Hikari.MaterialScene(primitives)
    scene = Hikari.Scene([lights...], bvh)
    return scene, camera, film
end

function render_whitted(mscene::Makie.Scene; samples_per_pixel=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.WhittedIntegrator(camera[], Hikari.UniformSampler(samples_per_pixel), max_depth)
    Hikari.integrator_threaded(integrator, scene, film, camera[])
    return film.framebuffer
end

function render_sppm(mscene::Makie.Scene; search_radius=0.075f0, max_depth=5, iterations=100)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.SPPMIntegrator(camera[], search_radius, max_depth, iterations, film)
    integrator(scene, film)
    return reverse(film.framebuffer, dims=1)
end

function render_gpu(mscene::Makie.Scene, ArrayType; samples_per_pixel=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.WhittedIntegrator(camera[], Hikari.UniformSampler(samples_per_pixel), max_depth)
    gpu_scene = Hikari.to_gpu(ArrayType, scene)
    gpu_film = Hikari.to_gpu(ArrayType, film)
    integrator(gpu_scene, gpu_film, camera[])
    return Array(film.framebuffer)
end


function render_interactive(mscene::Makie.Scene; backend, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    imsub = Scene(mscene)
    imgp = image!(imsub, -1 .. 1, -1 .. 1, film.framebuffer, uv_transform=(:rotr90, :flip_y))
    integrator = Hikari.WhittedIntegrator(camera[], Hikari.UniformSampler(1), max_depth)
    display(mscene; backend=backend)
    cam_start = camera[]
    loki = Threads.ReentrantLock()
    cam_rendered = camera[]
    Base.errormonitor(Threads.@spawn while !Makie.isclosed(mscene)
        if cam_rendered != camera[]
            cam_rendered = camera[]
            Hikari.clear!(film)
            imgp.visible = false
        end
        @time Hikari.integrator_threaded(integrator, scene, film, camera[])
        lock(loki) do
            imgp[3] = film.framebuffer
            imgp.visible = true
        end
        sleep(1/60)
    end)
    Base.errormonitor(Threads.@spawn while !Makie.isclosed(mscene)
        lock(loki) do
            if cam_start != camera[]
                cam_start = camera[]
                imgp.visible = false
            end
        end
        sleep(1/60)
    end)
    return
end

# re-export Makie, including deprecated names
for name in names(Makie, all=true)
    if Base.isexported(Makie, name)
        @eval using Makie: $(name)
        @eval export $(name)
    end
end

end
