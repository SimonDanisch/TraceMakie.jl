module TraceMakie

using Makie, Trace, Colors, LinearAlgebra, GeometryBasics

function to_spectrum(data::Colorant)
    rgb = RGBf(data)
    return Trace.RGBSpectrum(rgb.r, rgb.g, rgb.b)
end

function to_spectrum(data::AbstractMatrix{<:Colorant})
    colors = convert(AbstractMatrix{RGBf}, data)
    return collect(reinterpret(Trace.RGBSpectrum, colors))
end

function extract_material(plot::Plot, tex::Union{Trace.Texture, Nothing})
    if haskey(plot, :material) && !isnothing(to_value(plot.material))
        if to_value(plot.material) isa Trace.Material
            return to_value(plot.material)
        end
    elseif tex isa Nothing
        error("Neither color nor material are defined for plot: $plot")
    else
        return Trace.MatteMaterial(tex, Trace.ConstantTexture(0.0f0))
    end
end

function extract_material(plot::Plot, color_obs::Observable)
    color = to_value(color_obs)
    tex = nothing
    if color isa AbstractMatrix{<:Number}
        calc_color = to_value(plot.calculated_colors)
        tex = Trace.Texture(to_spectrum(to_color(calc_color)))
        onany(plot, color_obs, plot.colormap, plot.colorrange) do color, cmap, crange
            tex.data = to_spectrum(to_color(calc_color))
            return
        end
    elseif color isa AbstractMatrix{<:Colorant}
        tex = Trace.Texture(to_spectrum(color))
        onany(plot, color_obs) do color
            tex.data = to_spectrum(color)
            return
        end
    elseif color isa Colorant || color isa Union{String,Symbol}
        tex = Trace.ConstantTexture(to_spectrum(to_color(color)))
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
    triangles = Trace.create_triangle_mesh(plot.mesh[])
    material = extract_material(plot, plot.color)
    return Trace.GeometricPrimitive(triangles, material)
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
    # with this we can beuild a mesh
    mesh = normal_mesh(GeometryBasics.Mesh(meta(vec(positions[]), uv=uv), faces))

    triangles = Trace.create_triangle_mesh(mesh)
    material = extract_material(plot, plot.z)
    return Trace.GeometricPrimitive(triangles, material)
end

function to_trace_primitive(plot::Makie.Plot)
    return nothing
end

function to_trace_light(light::Makie.AmbientLight)
    return Trace.AmbientLight(
        to_spectrum(light.color[]),
    )
end

function to_trace_light(light::Makie.PointLight)
    return Trace.PointLight(
        Trace.translate(light.position[]), to_spectrum(light.color[]),
    )
end

function to_trace_light(light)
    return nothing
end

function to_trace_camera(scene::Makie.Scene, film)
    cc = scene.camera_controls
    return lift(scene, cc.eyeposition, cc.lookat, cc.upvector, cc.fov) do eyeposition, lookat, upvector, fov
        view = Trace.look_at(
            Point3f(eyeposition), Point3f(lookat), Vec3f(upvector),
        )
        return Trace.PerspectiveCamera(
            view, Trace.Bounds2(Point2f(-1.0f0), Point2f(1.0f0)),
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
    f = Trace.LanczosSincFilter(Point2f(1.0f0), 3.0f0)
    film = Trace.Film(
        resolution,
        Trace.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
    )
    primitives = []
    for plot in scene.plots
        prim = to_trace_primitive(plot)
        !isnothing(prim) && push!(primitives, prim)
    end
    camera = to_trace_camera(scene, film)
    lights = []
    for light in scene.lights
        l = to_trace_light(light)
        isnothing(l) || push!(lights, l)
    end
    if isempty(lights)
        error("Must have at least one light")
    end
    bvh = Trace.BVHAccel(primitives, 1)
    scene = Trace.Scene([lights...], bvh)
    return scene, camera, film
end

function render_whitted(mscene::Makie.Scene; samples_per_pixel=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Trace.WhittedIntegrator(camera[], Trace.UniformSampler(samples_per_pixel), max_depth)
    integrator(scene, film, camera[])
    return film.framebuffer
end

function render_sppm(mscene::Makie.Scene; search_radius=0.075f0, max_depth=5, iterations=100)
    scene, camera, film = convert_scene(mscene)
    integrator = Trace.SPPMIntegrator(camera[], search_radius, max_depth, iterations, film)
    integrator(scene, film)
    return reverse(film.framebuffer, dims=1)
end

function render_gpu(mscene::Makie.Scene, ArrayType; samples_per_pixel=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Trace.WhittedIntegrator(camera[], Trace.UniformSampler(samples_per_pixel), max_depth)
    preserve = []
    gpu_scene = Trace.to_gpu(ArrayType, scene; preserve=preserve)
    gpu_film = Trace.to_gpu(ArrayType, film; preserve=preserve)
    GC.@preserve preserve begin
        integrator(gpu_scene, gpu_film)
    end
    return Array(film.framebuffer)
end


function render_interactive(mscene::Makie.Scene; backend, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    imsub = Scene(mscene)
    imgp = image!(imsub, -1 .. 1, -1 .. 1, film.framebuffer, uv_transform=(:rotr90, :flip_x))
    integrator = Trace.WhittedIntegrator(camera[], Trace.UniformSampler(1), max_depth)
    display(mscene; backend=backend)
    cam_start = camera[]
    Base.errormonitor(@async while isopen(mscene)
        if cam_start != camera[]
            cam_start = camera[]
            Trace.clear!(film)
            imgp.visible = false
        end
        integrator(scene, film, camera[])
        imgp[3] = film.framebuffer
        imgp.visible = true
        yield()
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
