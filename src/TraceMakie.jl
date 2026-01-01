module TraceMakie

using Makie, Hikari, Colors, LinearAlgebra, GeometryBasics, Raycore
using Makie: Observable, on, colorbuffer, to_value
using Makie: Quaternionf
using GeometryBasics: VecTypes
using Colors: N0f8
import Makie.Observables

# =============================================================================
# ScreenConfig
# =============================================================================

# Convenience constructors for integrators
"""
    Whitted(; samples_per_pixel=8, max_depth=5)

Create a Whitted-style ray tracing integrator. Fast but only handles direct lighting
and perfect specular reflections/refractions.
"""
Whitted(; samples_per_pixel=8, max_depth=5) = Hikari.WhittedIntegrator(Hikari.UniformSampler(samples_per_pixel), max_depth)

"""
    ScreenConfig

Configuration for TraceMakie rendering.

* `integrator`: The integrator to use for rendering (default: `Whitted()`)
  - `Whitted(; samples_per_pixel=8, max_depth=5)` - Fast Whitted-style ray tracing
* `exposure`: Exposure multiplier for postprocessing (default: 1.0)
* `tonemap`: Tonemapping method (default: :aces)
  - `:reinhard` - Simple Reinhard L/(1+L)
  - `:reinhard_extended` - Extended Reinhard with white point
  - `:aces` - ACES filmic (industry standard)
  - `:uncharted2` - Uncharted 2 filmic
  - `:filmic` - Hejl-Dawson filmic
  - `nothing` - No tonemapping (linear clamp)
* `gamma`: Gamma correction value (default: 2.2, use `nothing` to skip)
"""
struct ScreenConfig
    integrator::Hikari.SamplerIntegrator
    exposure::Float32
    tonemap::Union{Symbol, Nothing}
    gamma::Union{Float32, Nothing}

    function ScreenConfig(integrator, exposure, tonemap, gamma)
        actual_integrator = integrator isa Makie.Automatic ? Whitted() : integrator
        actual_exposure = Float32(exposure)
        actual_gamma = isnothing(gamma) ? nothing : Float32(gamma)
        return new(actual_integrator, actual_exposure, tonemap, actual_gamma)
    end
end

# =============================================================================
# TraceMakieState: Tracks the mapping between Makie plots and Hikari instances
# =============================================================================

"""
    PlotInfo

Stores information about a single Makie plot in the ray tracing scene.
For MeshScatter plots, `instance_count` tracks the number of instances sharing one BLAS.
"""
mutable struct PlotInfo
    plot::Makie.AbstractPlot
    handle::Raycore.InstanceHandle
    transform_obs::Union{Observable, Nothing}
    obs_funcs::Vector{Observables.ObserverFunction}
    instance_count::Int  # Number of instances (>1 for MeshScatter)
    per_instance_materials::Bool  # True if each instance has separate material (no batched transforms)
    first_instance_idx::Int  # Starting index in TLAS.instances for per-instance materials
end

PlotInfo(plot, handle, transform_obs, obs_funcs) = PlotInfo(plot, handle, transform_obs, obs_funcs, 1, false, 0)
PlotInfo(plot, handle, transform_obs, obs_funcs, count) = PlotInfo(plot, handle, transform_obs, obs_funcs, count, false, 0)
PlotInfo(plot, handle, transform_obs, obs_funcs, count, per_inst, first_idx) = PlotInfo(plot, handle, transform_obs, obs_funcs, count, per_inst, first_idx)

"""
    TraceMakieState

Holds the state needed to synchronize a Makie scene with a Hikari ray tracing scene.
Supports dynamic updates to transformations via TLAS refit.
"""
mutable struct TraceMakieState
    tlas::Raycore.TLAS
    materials::Tuple  # Tuple of material vectors (for MaterialScene)
    plot_infos::Vector{PlotInfo}
    lights::Tuple  # Tuple of lights (type-stable for rendering)
    film::Hikari.Film
    camera::Observable
    needs_refit::Bool  # Flag to track if TLAS needs refit
    hikari_scene::Hikari.Scene  # Cached scene to avoid recreating on each render
end

# =============================================================================
# Screen
# =============================================================================

"""
    Screen <: Makie.MakieScreen

TraceMakie screen for ray-traced rendering.

# Constructors

    Screen(scene::Scene; screen_config...)
    Screen(scene::Scene, config::ScreenConfig)

# Configuration options (via screen_config or ScreenConfig):

$(Base.doc(ScreenConfig))
"""
mutable struct Screen <: Makie.MakieScreen
    scene::Union{Nothing, Scene}
    state::Union{Nothing, TraceMakieState}
    config::ScreenConfig
end

Base.isopen(::Screen) = true
Base.size(screen::Screen) = isnothing(screen.scene) ? (0, 0) : size(screen.scene)

function Screen(fb_size::NTuple{2, <:Integer}; screen_config...)
    config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol, Any}(screen_config))
    return Screen(fb_size, config)
end

function Screen(::NTuple{2, <:Integer}, config::ScreenConfig)
    return Screen(nothing, nothing, config)
end

function Screen(scene::Scene; screen_config...)
    config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol, Any}(screen_config))
    return Screen(scene, config)
end

function Screen(scene::Scene, config::ScreenConfig)
    screen = Screen(size(scene), config)
    screen.scene = scene
    return screen
end

Screen(scene::Scene, config::ScreenConfig, ::IO, ::MIME) = Screen(scene, config)
Screen(scene::Scene, config::ScreenConfig, ::Makie.ImageStorageFormat) = Screen(scene, config)

Makie.apply_screen_config!(screen::Screen, ::ScreenConfig, args...) = screen
Base.empty!(::Screen) = nothing

# =============================================================================
# Rendering
# =============================================================================

function render!(screen::Screen)
    state = screen.state
    scene = screen.scene
    isnothing(state) && error("Screen not set up - call display first")
    isnothing(scene) && error("No scene attached to screen")

    # Sync transforms and refit TLAS if needed
    sync_transforms!(state)

    # Clear film
    Hikari.clear!(state.film)

    # Render using integrator from config
    camera = state.camera[]
    screen.config.integrator(state.hikari_scene, state.film, camera)

    return state.film.framebuffer
end

function Makie.colorbuffer(screen::Screen, format::Makie.ImageStorageFormat = Makie.JuliaNative; figure = nothing)
    if isnothing(screen.state)
        display(screen, screen.scene; figure = figure)
    end

    render!(screen)

    # Apply postprocessing (tonemapping, gamma, exposure)
    config = screen.config
    Hikari.postprocess!(screen.state.film;
        exposure = config.exposure,
        tonemap = config.tonemap,
        gamma = config.gamma
    )

    # Convert postprocessed buffer to RGB{N0f8}
    result = map(screen.state.film.postprocess) do c
        RGB{N0f8}(c.r, c.g, c.b)
    end

    if format == Makie.GLNative
        return Makie.jl_to_gl_format(result)
    else # JuliaNative
        return result
    end
end

"""
    postprocess!(screen::Screen; exposure=nothing, tonemap=nothing, gamma=nothing)

Re-apply postprocessing to an already-rendered screen without re-rendering.

This is useful for quickly experimenting with different postprocessing settings
after a render is complete. Parameters that are not specified will use the
screen's existing config values.

# Arguments
- `screen`: A Screen that has already been rendered
- `exposure`: Exposure multiplier (default: use screen config)
- `tonemap`: Tonemapping method (:aces, :reinhard, :uncharted2, :filmic, or nothing)
- `gamma`: Gamma correction value (default: use screen config)

# Returns
The postprocessed image as `Matrix{RGB{N0f8}}`

# Example
```julia
# Render once
screen = TraceMakie.Screen(scene)
img = Makie.colorbuffer(screen)

# Try different postprocessing without re-rendering
img_bright = TraceMakie.postprocess!(screen; exposure=2.0)
img_filmic = TraceMakie.postprocess!(screen; tonemap=:filmic)
img_low_gamma = TraceMakie.postprocess!(screen; gamma=1.8)
```
"""
function postprocess!(screen::Screen;
    exposure::Union{Real, Nothing} = nothing,
    tonemap::Union{Symbol, Nothing, Missing} = missing,  # missing = use config, nothing = no tonemap
    gamma::Union{Real, Nothing} = nothing,
)
    if isnothing(screen.state)
        error("Screen has not been rendered yet. Call Makie.colorbuffer(screen) first.")
    end

    # Use provided values or fall back to screen config
    exp_val = isnothing(exposure) ? screen.config.exposure : Float32(exposure)
    tm_val = ismissing(tonemap) ? screen.config.tonemap : tonemap
    gamma_val = isnothing(gamma) ? screen.config.gamma : Float32(gamma)

    # Apply postprocessing
    Hikari.postprocess!(screen.state.film;
        exposure = exp_val,
        tonemap = tm_val,
        gamma = gamma_val
    )

    # Convert to RGB{N0f8}
    result = map(screen.state.film.postprocess) do c
        RGB{N0f8}(c.r, c.g, c.b)
    end

    return result
end

function Base.display(screen::Screen, scene::Scene; figure = nothing, display_kw...)
    screen.scene = scene
    screen.state = convert_scene_with_state(scene)
    return screen
end

function Base.insert!(screen::Screen, scene::Scene, plot::AbstractPlot)
    # For now, rebuild the entire state when plots change
    # Future: incremental updates
    if !isnothing(screen.state)
        screen.state = convert_scene_with_state(scene)
    end
    return screen
end

Makie.backend_showable(::Type{Screen}, ::Union{MIME"image/jpeg", MIME"image/png"}) = true

# =============================================================================
# Backend activation
# =============================================================================

"""
    TraceMakie.activate!(; screen_config...)

Sets TraceMakie as the currently active backend and allows setting screen configuration.

# Arguments (via screen_config):

$(Base.doc(ScreenConfig))

# Examples

```julia
# Use default Whitted integrator
TraceMakie.activate!()

# Use Whitted with custom settings
TraceMakie.activate!(integrator = TraceMakie.Whitted(samples_per_pixel=16, max_depth=8))

# Configure postprocessing
TraceMakie.activate!(exposure = 1.5, tonemap = :reinhard, gamma = 2.2)
```
"""
function activate!(; screen_config...)
    Makie.set_screen_config!(TraceMakie, screen_config)
    Makie.set_active_backend!(TraceMakie)
    return
end

function __init__()
    # Activate TraceMakie as the default backend when loaded
    activate!()
    return
end

"""
    get_plot_transform(plot) -> Mat4f

Extract the full transformation matrix from a Makie plot.
"""
function get_plot_transform(plot::Makie.AbstractPlot)
    return Mat4f(Makie.transformationmatrix(plot)[])
end

"""
    update_plot_transform!(state::TraceMakieState, info::PlotInfo)

Update a single plot's transform in the TLAS.
"""
function update_plot_transform!(state::TraceMakieState, info::PlotInfo)
    transform = get_plot_transform(info.plot)
    Raycore.update_transform!(state.tlas, info.handle, transform)
    state.needs_refit = true
end

"""
    refit_if_needed!(state::TraceMakieState)

Refit the TLAS if any transforms have changed.
"""
function refit_if_needed!(state::TraceMakieState)
    if state.needs_refit
        Raycore.refit_tlas!(state.tlas)
        state.needs_refit = false
    end
end

# =============================================================================
# Color/Spectrum conversion
# =============================================================================

function to_spectrum(data::Colorant)
    rgb = RGBf(data)
    alpha = data isa TransparentColor ? Float32(Colors.alpha(data)) : 1f0
    return Hikari.RGBSpectrum(rgb.r, rgb.g, rgb.b, alpha)
end

function to_spectrum(data::AbstractMatrix{<:Colorant})
    return map(data) do c
        rgb = RGBf(c)
        alpha = c isa TransparentColor ? Float32(Colors.alpha(c)) : 1f0
        Hikari.RGBSpectrum(rgb.r, rgb.g, rgb.b, alpha)
    end
end

"""
    merge_color_with_material(color_tex::Hikari.Texture, material::Hikari.Material)

Create a new material of the same type but with the color texture merged in.
The color modulates the material's primary color channel (Kd, Kr, etc.).
"""
function merge_color_with_material(color_tex::Hikari.Texture, material::Hikari.MatteMaterial)
    Hikari.MatteMaterial(color_tex, material.σ)
end

function merge_color_with_material(color_tex::Hikari.Texture, material::Hikari.MirrorMaterial)
    Hikari.MirrorMaterial(color_tex)
end

function merge_color_with_material(color_tex::Hikari.Texture, material::Hikari.GlassMaterial)
    # Use color for transmittance (Kt) - this tints the glass
    Hikari.GlassMaterial(
        material.Kr, color_tex,
        material.u_roughness, material.v_roughness,
        material.index, material.remap_roughness
    )
end

function merge_color_with_material(color_tex::Hikari.Texture, material::Hikari.PlasticMaterial)
    Hikari.PlasticMaterial(color_tex, material.Ks, material.roughness, material.remap_roughness)
end

function merge_color_with_material(color_tex::Hikari.Texture, material::Hikari.MetalMaterial)
    # For metal, color is used as a reflectance tint that multiplies the Fresnel result
    # This preserves the physical eta/k values while allowing color variation
    Hikari.MetalMaterial(material.eta, material.k, material.roughness, color_tex, material.remap_roughness)
end

# Fallback for unknown material types - just return the material as-is
function merge_color_with_material(color_tex::Hikari.Texture, material::Hikari.Material)
    @warn "Unknown material type $(typeof(material)), ignoring color"
    material
end

function extract_material(plot::Plot, tex::Union{Hikari.Texture, Nothing})
    has_material = haskey(plot, :material) && !isnothing(to_value(plot.material))
    material = has_material ? to_value(plot.material) : nothing

    if material isa Hikari.Material && tex isa Hikari.Texture
        # Both color and material provided - merge them
        return merge_color_with_material(tex, material)
    elseif material isa Hikari.Material
        # Only material provided - use as-is
        return material
    elseif tex isa Hikari.Texture
        # Only color provided - create MatteMaterial
        return Hikari.MatteMaterial(tex, Hikari.ConstantTexture(0.0f0))
    else
        error("Neither color nor material are defined for plot: $plot")
    end
end

function extract_material(plot::Plot, color_obs::Union{Makie.Computed, Observable})
    color = to_value(color_obs)

    # Check if material is explicitly provided
    has_material = haskey(plot, :material) && !isnothing(to_value(plot.material))
    material = has_material ? to_value(plot.material) : nothing

    # If material is provided and color is the default (not explicitly set by user),
    # just use the material as-is without merging
    if material isa Hikari.Material && color isa Colorant
        # Check if this looks like Makie's default color (blue) - if so, ignore it
        c = to_color(color)
        is_default_blue = (red(c) ≈ 0.0f0) && (green(c) ≈ 0.447f0) && (blue(c) ≈ 0.698f0)
        if is_default_blue
            return material
        end
    end

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
    elseif color isa AbstractVector{<:Colorant}
        # Per-instance colors (e.g., for meshscatter)
        # Convert to spectrum array for per-instance materials
        tex = Hikari.Texture(to_spectrum.(color))
        onany(plot, color_obs) do color
            tex.data = to_spectrum.(color)
            return
        end
    elseif color isa Colorant || color isa Union{String,Symbol}
        tex = Hikari.ConstantTexture(to_spectrum(to_color(color)))
    elseif color isa Nothing
        # ignore!
        nothing
    else
        error("Unsupported color type for TraceMakie backend: $(typeof(color))")
    end

    return extract_material(plot, tex)
end

"""
Convert a Makie material dict (from GLB) to a Hikari material.
"""
function glb_material_to_hikari(mat_dict::Dict{String, Any})
    # Check for diffuse map (texture)
    if haskey(mat_dict, "diffuse map")
        diffuse_map = mat_dict["diffuse map"]
        if haskey(diffuse_map, "image")
            img = diffuse_map["image"]
            tex = Hikari.Texture(to_spectrum(img))
            roughness = get(mat_dict, "roughness", 0.5f0)
            return Hikari.MatteMaterial(tex, Hikari.ConstantTexture(Float32(roughness) * 90f0))
        end
    end

    # Check for diffuse color
    if haskey(mat_dict, "diffuse")
        diffuse = mat_dict["diffuse"]
        color = RGBf(diffuse[1], diffuse[2], diffuse[3])
        tex = Hikari.ConstantTexture(to_spectrum(color))
        roughness = get(mat_dict, "roughness", 0.5f0)
        return Hikari.MatteMaterial(tex, Hikari.ConstantTexture(Float32(roughness) * 90f0))
    end

    # Default: white matte
    return Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.8f0, 0.8f0, 0.8f0)),
        Hikari.ConstantTexture(0.0f0)
    )
end

function to_trace_primitive(plot::Makie.Mesh)
    mesh = plot.mesh[]

    # Handle MetaMesh with materials
    if mesh isa GeometryBasics.MetaMesh
        primitives = Tuple[]

        # Check if we have material info
        if haskey(mesh, :material_names) && haskey(mesh, :materials)
            submeshes = GeometryBasics.split_mesh(mesh.mesh)
            material_names = mesh[:material_names]
            materials_dict = mesh[:materials]

            # Cache converted materials to avoid creating duplicate textures
            hikari_materials = Dict{String, Any}()
            default_mat = nothing

            for (name, submesh) in zip(material_names, submeshes)
                tmesh = Raycore.TriangleMesh(submesh)

                # Get or create cached material
                mat = get!(hikari_materials, name) do
                    if haskey(materials_dict, name)
                        glb_material_to_hikari(materials_dict[name])
                    else
                        if isnothing(default_mat)
                            default_mat = extract_material(plot, plot.color)
                        end
                        default_mat
                    end
                end

                push!(primitives, (tmesh, mat))
            end
        else
            # MetaMesh without material info - treat as single mesh
            tmesh = Raycore.TriangleMesh(mesh.mesh)
            mat = extract_material(plot, plot.color)
            push!(primitives, (tmesh, mat))
        end

        return primitives
    end

    # Regular mesh
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

"""
    build_materials_tuple(materials_list::Vector{<:Hikari.Material}) -> Tuple

Group materials by type into a tuple of vectors for MaterialScene.
"""
function build_materials_tuple(materials_list::Vector)
    if isempty(materials_list)
        return (Hikari.MatteMaterial[],)
    end

    # Group by type
    type_to_materials = Dict{DataType, Vector}()
    type_order = DataType[]

    for mat in materials_list
        T = typeof(mat)
        if !haskey(type_to_materials, T)
            type_to_materials[T] = T[]
            push!(type_order, T)
        end
        push!(type_to_materials[T], mat)
    end

    # Build tuple in order
    return Tuple([type_to_materials[T] for T in type_order])
end

"""
    convert_scene_with_state(scene::Makie.Scene) -> TraceMakieState

Convert a Makie scene to a TraceMakieState that supports dynamic transform updates.
Automatically watches plot transformations and syncs to TLAS.
"""
function convert_scene_with_state(mscene::Makie.Scene)
    resolution = Point2f(size(mscene))
    f = Hikari.LanczosSincFilter(Point2f(1.0f0), 3.0f0)
    film = Hikari.Film(
        resolution,
        Hikari.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
    )

    # Collect Instance objects and materials
    # MeshScatter creates a single Instance with multiple transforms (efficient instancing)
    # Regular meshes create one Instance per mesh
    instances = Raycore.Instance[]
    materials_list = Hikari.Material[]
    plot_to_instance_info = Dict{Makie.AbstractPlot, Tuple{Int, Int, Bool, Int}}()  # plot -> (first_instance_idx, count, per_instance_materials, first_descriptor_idx)

    # Helper to get or add material and return (type_slot, index_within_type)
    # We track materials grouped by type for proper MaterialIndex
    type_to_slot = Dict{DataType, UInt8}()
    type_to_materials = Dict{DataType, Vector{Hikari.Material}}()
    type_order = DataType[]

    function get_material_index(mat::Hikari.Material)
        T = typeof(mat)
        if !haskey(type_to_slot, T)
            type_to_slot[T] = UInt8(length(type_to_slot) + 1)
            type_to_materials[T] = Hikari.Material[]
            push!(type_order, T)
        end
        slot = type_to_slot[T]
        # Check if this exact material already exists
        existing_idx = findfirst(==(mat), type_to_materials[T])
        if !isnothing(existing_idx)
            return Hikari.MaterialIndex(slot, UInt32(existing_idx))
        end
        # Add new material
        push!(type_to_materials[T], mat)
        push!(materials_list, mat)
        return Hikari.MaterialIndex(slot, UInt32(length(type_to_materials[T])))
    end

    # Track cumulative InstanceDescriptor count (not Instance count)
    # because one Instance with N transforms creates N InstanceDescriptors
    total_instance_descriptors = 0

    for plot in mscene.plots
        result = to_trace_primitive_with_transform(plot)
        if !isnothing(result)
            if result isa MeshScatterResult
                first_idx = length(instances) + 1
                first_descriptor_idx = total_instance_descriptors + 1
                n_instances = length(result.transforms)

                has_per_instance_mats = result.materials isa Vector
                if has_per_instance_mats
                    # Per-instance materials: create separate Instance for each
                    # This creates one BLAS per instance (less efficient but correct)
                    for (transform, mat) in zip(result.transforms, result.materials)
                        mat_index = get_material_index(mat)
                        push!(instances, Raycore.Instance(result.mesh, transform, mat_index))
                    end
                    total_instance_descriptors += n_instances
                else
                    # Single material for all instances (efficient instancing)
                    mat_index = get_material_index(result.materials)
                    metadata = [mat_index for _ in 1:n_instances]
                    push!(instances, Raycore.Instance(result.mesh, result.transforms, metadata))
                    total_instance_descriptors += n_instances
                end

                plot_to_instance_info[plot] = (first_idx, n_instances, has_per_instance_mats, first_descriptor_idx)
            elseif result isa Vector
                # Multiple primitives from MetaMesh - each gets its own Instance
                first_idx = length(instances) + 1
                first_descriptor_idx = total_instance_descriptors + 1
                for (mesh, mat, transform) in result
                    mat_index = get_material_index(mat)
                    push!(instances, Raycore.Instance(mesh, transform, mat_index))
                end
                total_instance_descriptors += length(result)
                plot_to_instance_info[plot] = (first_idx, length(result), false, first_descriptor_idx)
            else
                mesh, mat, transform = result
                first_idx = length(instances) + 1
                first_descriptor_idx = total_instance_descriptors + 1
                mat_index = get_material_index(mat)
                push!(instances, Raycore.Instance(mesh, transform, mat_index))
                total_instance_descriptors += 1
                plot_to_instance_info[plot] = (first_idx, 1, false, first_descriptor_idx)
            end
        end
    end

    # Build TLAS from instances
    tlas, handles = Raycore.TLAS(instances)

    # Build materials tuple from type-grouped materials
    materials = if isempty(type_order)
        (Hikari.MatteMaterial[],)
    else
        Tuple([Vector{T}(type_to_materials[T]) for T in type_order])
    end

    # Create PlotInfos
    plot_infos = PlotInfo[]
    for (plot, (first_idx, count, per_instance_mats, first_descriptor_idx)) in plot_to_instance_info
        handle = handles[first_idx]
        transform_obs = Makie.transformationmatrix(plot)
        obs_funcs = Observables.ObserverFunction[]
        # For per-instance materials, we track the starting InstanceDescriptor index
        # because each instance has a different blas_index
        info = PlotInfo(plot, handle, transform_obs, obs_funcs, count, per_instance_mats, first_descriptor_idx)
        push!(plot_infos, info)
    end

    camera = to_trace_camera(mscene, film)

    # Extract lights
    lights = Hikari.Light[]
    makie_lights = Makie.get_lights(mscene)
    for light in makie_lights
        l = to_trace_light(light)
        isnothing(l) || push!(lights, l)
    end

    # Add ambient light if present
    if haskey(mscene.compute, :ambient_color)
        ambient_color = mscene.compute[:ambient_color][]
        if ambient_color != RGBf(0, 0, 0)
            push!(lights, Hikari.AmbientLight(to_spectrum(ambient_color)))
        end
    end

    if isempty(lights)
        error("Must have at least one light")
    end

    # Convert lights to tuple for type stability
    lights_tuple = Tuple(lights)

    # Create cached hikari scene
    material_scene = Hikari.MaterialScene(tlas, materials)
    hikari_scene = Hikari.Scene(lights, material_scene)

    state = TraceMakieState(tlas, materials, plot_infos, lights_tuple, film, camera, false, hikari_scene)

    # Register transform observers
    for info in plot_infos
        obs_func = on(info.transform_obs) do _
            update_plot_transform!(state, info)
        end
        push!(info.obs_funcs, obs_func)
    end

    return state
end

"""
    to_trace_primitive_with_transform(plot) -> (mesh, material, transform) or Vector or nothing

Like to_trace_primitive but also extracts the plot's transformation matrix.
"""
function to_trace_primitive_with_transform(plot::Makie.Mesh)
    mesh = plot.mesh[]
    transform = get_plot_transform(plot)

    # Handle MetaMesh with materials
    if mesh isa GeometryBasics.MetaMesh
        results = []

        if haskey(mesh, :material_names) && haskey(mesh, :materials)
            submeshes = GeometryBasics.split_mesh(mesh.mesh)
            material_names = mesh[:material_names]
            materials_dict = mesh[:materials]

            hikari_materials = Dict{String, Any}()
            default_mat = nothing

            for (name, submesh) in zip(material_names, submeshes)
                tmesh = Raycore.TriangleMesh(submesh)

                mat = get!(hikari_materials, name) do
                    if haskey(materials_dict, name)
                        glb_material_to_hikari(materials_dict[name])
                    else
                        if isnothing(default_mat)
                            default_mat = extract_material(plot, plot.color)
                        end
                        default_mat
                    end
                end

                push!(results, (tmesh, mat, transform))
            end
        else
            tmesh = Raycore.TriangleMesh(mesh.mesh)
            mat = extract_material(plot, plot.color)
            push!(results, (tmesh, mat, transform))
        end

        return results
    end

    # Regular mesh
    tmesh = Raycore.TriangleMesh(mesh)
    material = extract_material(plot, plot.color)
    return (tmesh, material, transform)
end

function to_trace_primitive_with_transform(plot::Makie.Surface)
    # Surface doesn't support transforms well, fall back to identity
    prim = to_trace_primitive(plot)
    if isnothing(prim)
        return nothing
    end
    # Extract mesh and material from GeometricPrimitive
    return (prim.shape, prim.material, Mat4f(I))
end

function to_trace_primitive_with_transform(plot::Makie.Plot)
    return nothing
end

# =============================================================================
# MeshScatter support - efficient instancing with TLAS
# =============================================================================

"""
    meshscatter_marker_mesh(marker)

Convert a MeshScatter marker to a mesh. Handles geometry primitives and meshes.
"""
function meshscatter_marker_mesh(marker)
    if marker isa GeometryBasics.Mesh
        return marker
    elseif marker isa GeometryBasics.GeometryPrimitive
        return GeometryBasics.normal_mesh(marker)
    elseif marker == :Sphere || marker === Makie.automatic
        return GeometryBasics.normal_mesh(GeometryBasics.Sphere(Point3f(0), 1.0f0))
    elseif marker isa Symbol
        # Try to get a builtin marker
        return GeometryBasics.normal_mesh(Makie.default_marker_map()[marker])
    else
        error("Unsupported MeshScatter marker type: $(typeof(marker))")
    end
end

"""
    meshscatter_transforms(positions, markersize, rotation, plot_transform)

Build per-instance transform matrices for MeshScatter.
Each instance gets: plot_transform * translate(position) * scale(markersize) * rotate(rotation)
"""
function meshscatter_transforms(positions, markersize, rotation, plot_transform::Mat4f)
    n = length(positions)

    # Normalize markersize to per-instance Vec3f
    scales = if markersize isa Number
        fill(Vec3f(markersize), n)
    elseif markersize isa VecTypes{3}
        fill(Vec3f(markersize), n)
    elseif markersize isa AbstractVector
        if eltype(markersize) <: Number
            [Vec3f(s) for s in markersize]
        else
            [Vec3f(s) for s in markersize]
        end
    else
        fill(Vec3f(0.1f0), n)  # Default markersize
    end

    # Normalize rotation to per-instance Quaternion
    rotations = if rotation isa Quaternionf
        fill(rotation, n)
    elseif rotation isa Number
        # Rotation around z-axis
        q = Makie.qrotation(Vec3f(0, 0, 1), Float32(rotation))
        fill(q, n)
    elseif rotation isa VecTypes{3}
        # Vec3f interpreted as axis to align z-axis with
        q = Makie.rotation_between(Vec3f(0, 0, 1), Vec3f(rotation))
        fill(q, n)
    elseif rotation isa AbstractVector
        [rotation_to_quaternion(r) for r in rotation]
    else
        fill(Quaternionf(0, 0, 0, 1), n)
    end

    # Build transform matrices
    transforms = Mat4f[]
    for i in 1:n
        pos = positions[i]
        s = scales[min(i, length(scales))]
        r = rotations[min(i, length(rotations))]

        # Build local transform: T * S * R
        # Translation matrix
        T = Mat4f(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            pos[1], pos[2], pos[3], 1
        )
        # Scale matrix
        S = Mat4f(
            s[1], 0, 0, 0,
            0, s[2], 0, 0,
            0, 0, s[3], 0,
            0, 0, 0, 1
        )
        # Rotation matrix from quaternion
        R = Mat4f(Makie.rotationmatrix4(r))

        # Combine: plot_transform * T * R * S
        local_transform = T * R * S
        push!(transforms, plot_transform * local_transform)
    end

    return transforms
end

"""Helper to convert various rotation types to Quaternion."""
function rotation_to_quaternion(r)
    if r isa Quaternionf
        return r
    elseif r isa Number
        return Makie.qrotation(Vec3f(0, 0, 1), Float32(r))
    elseif r isa VecTypes{3}
        return Makie.rotation_between(Vec3f(0, 0, 1), Vec3f(r))
    else
        return Quaternionf(0, 0, 0, 1)
    end
end

"""
    to_trace_primitive_with_transform(plot::Makie.MeshScatter) -> MeshScatterResult

Returns a special result type for MeshScatter with:
- mesh: The marker mesh (single BLAS)
- materials: Either a single material (for all instances) or Vector of per-instance materials
- transforms: Vector of per-instance transforms
"""
struct MeshScatterResult
    mesh::Any
    materials::Union{Hikari.Material, Vector{<:Hikari.Material}}
    transforms::Vector{Mat4f}
end

function to_trace_primitive_with_transform(plot::Makie.MeshScatter)
    # Get positions
    positions = to_value(plot.positions)
    if isempty(positions)
        return nothing
    end

    # Get marker mesh
    marker = to_value(plot.marker)
    mesh = meshscatter_marker_mesh(marker)
    tmesh = Raycore.TriangleMesh(mesh)

    # Get transform parameters
    markersize = to_value(plot.markersize)
    rotation = to_value(plot.rotation)
    plot_transform = get_plot_transform(plot)

    # Build per-instance transforms
    transforms = meshscatter_transforms(positions, markersize, rotation, plot_transform)

    # Get material(s)
    materials = extract_meshscatter_materials(plot, length(positions))

    return MeshScatterResult(tmesh, materials, transforms)
end

"""
Extract materials for meshscatter - returns either single material or per-instance materials.
"""
function extract_meshscatter_materials(plot::Makie.MeshScatter, n_instances::Int)
    color = to_value(plot.color)
    has_material = haskey(plot, :material) && !isnothing(to_value(plot.material))
    material_template = has_material ? to_value(plot.material) : nothing

    # Check if we have per-instance colors
    if color isa AbstractVector{<:Colorant} && length(color) == n_instances
        # Per-instance colors - create one material per instance
        return [create_material_with_color(to_color(c), material_template) for c in color]
    elseif color isa AbstractVector && length(color) == n_instances
        # Per-instance numeric values - use colormap
        calc_colors = to_value(plot.calculated_colors)
        if calc_colors isa AbstractVector{<:Colorant}
            return [create_material_with_color(to_color(c), material_template) for c in calc_colors]
        end
    end

    # Single material for all instances
    return extract_material(plot, plot.color)
end

"""
Create a material with the given color, optionally based on a template material.
"""
function create_material_with_color(color::Colorant, template::Nothing)
    # Default to MatteMaterial with the color
    Hikari.MatteMaterial(Hikari.ConstantTexture(to_spectrum(color)), Hikari.ConstantTexture(0.0f0))
end

function create_material_with_color(color::Colorant, template::Hikari.MatteMaterial)
    Hikari.MatteMaterial(Hikari.ConstantTexture(to_spectrum(color)), template.σ)
end

function create_material_with_color(color::Colorant, template::Hikari.PlasticMaterial)
    Hikari.PlasticMaterial(
        Hikari.ConstantTexture(to_spectrum(color)),
        template.Ks, template.roughness, template.remap_roughness
    )
end

function create_material_with_color(color::Colorant, template::Hikari.MetalMaterial)
    # For metals, color is used as reflectance tint (multiplies Fresnel result)
    # This preserves the physical eta/k values while allowing color variation
    Hikari.MetalMaterial(
        template.eta, template.k, template.roughness,
        Hikari.ConstantTexture(to_spectrum(color)),
        template.remap_roughness
    )
end

function create_material_with_color(color::Colorant, template::Hikari.Material)
    # Fallback: use MatteMaterial with the color
    @warn "Unsupported material type $(typeof(template)) for per-instance colors, using MatteMaterial"
    Hikari.MatteMaterial(Hikari.ConstantTexture(to_spectrum(color)), Hikari.ConstantTexture(0.0f0))
end

# Keep the old convert_scene for backwards compatibility
function convert_scene(mscene::Makie.Scene)
    state = convert_scene_with_state(mscene)
    return state.hikari_scene, state.camera, state.film
end

"""
    sync_transforms!(state::TraceMakieState)

Sync all plot transforms to the TLAS and refit.
Call this before rendering if transforms may have changed.
"""
function sync_transforms!(state::TraceMakieState)
    for info in state.plot_infos
        if info.instance_count > 1 && info.plot isa Makie.MeshScatter && !info.per_instance_materials
            # MeshScatter with shared material: update all instance transforms at once
            sync_meshscatter_transforms!(state, info)
        elseif info.instance_count > 1 && info.per_instance_materials
            # Per-instance materials: each instance is separate, update individually
            sync_meshscatter_transforms_individual!(state, info)
        else
            # Regular plot: single transform update
            transform = get_plot_transform(info.plot)
            Raycore.update_transform!(state.tlas, info.handle, transform)
        end
    end
    Raycore.refit_tlas!(state.tlas)
    state.needs_refit = false
end

"""
    sync_meshscatter_transforms!(state::TraceMakieState, info::PlotInfo)

Update all instance transforms for a MeshScatter plot with shared material.
"""
function sync_meshscatter_transforms!(state::TraceMakieState, info::PlotInfo)
    plot = info.plot
    positions = to_value(plot.positions)
    markersize = to_value(plot.markersize)
    rotation = to_value(plot.rotation)
    plot_transform = get_plot_transform(plot)

    transforms = meshscatter_transforms(positions, markersize, rotation, plot_transform)
    Raycore.update_transforms!(state.tlas, info.handle, transforms)
end

"""
    sync_meshscatter_transforms_individual!(state::TraceMakieState, info::PlotInfo)

Update transforms for a MeshScatter plot with per-instance materials.
Each instance is stored separately with its own BLAS, so we update them by index range.
"""
function sync_meshscatter_transforms_individual!(state::TraceMakieState, info::PlotInfo)
    plot = info.plot
    positions = to_value(plot.positions)
    markersize = to_value(plot.markersize)
    rotation = to_value(plot.rotation)
    plot_transform = get_plot_transform(plot)

    transforms = meshscatter_transforms(positions, markersize, rotation, plot_transform)

    # For per-instance materials, instances are stored consecutively starting at first_instance_idx
    # Each particle has its own BLAS, so we can't search by blas_index
    first_idx = info.first_instance_idx
    for i in 1:info.instance_count
        if i <= length(transforms)
            Raycore.update_instance_transform!(state.tlas, first_idx + i - 1, transforms[i])
        end
    end
end

"""
    render_frame!(state::TraceMakieState; samples_per_pixel=1, max_depth=5) -> Matrix

Render a single frame using the current state. Syncs transforms and refits TLAS if needed.
"""
function render_frame!(state::TraceMakieState; samples_per_pixel=1, max_depth=5)
    refit_if_needed!(state)
    Hikari.clear!(state.film)
    integrator = Hikari.WhittedIntegrator(Hikari.UniformSampler(samples_per_pixel), max_depth)
    integrator(state.hikari_scene, state.film, state.camera[])
    return state.film.framebuffer
end

function render_whitted(mscene::Makie.Scene; samples_per_pixel=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.WhittedIntegrator(Hikari.UniformSampler(samples_per_pixel), max_depth)
    # Call integrator directly - it uses KernelAbstractions for CPU/GPU dispatch
    integrator(scene, film, camera[])
    return film.framebuffer
end

function render_sppm(mscene::Makie.Scene; search_radius=0.075f0, max_depth=5, iterations=100)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.SPPMIntegrator(search_radius, max_depth, iterations, film)
    integrator(scene, film, camera[])
    return film.framebuffer
end

function render_gpu(mscene::Makie.Scene, ArrayType; samples_per_pixel=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.WhittedIntegrator(Hikari.UniformSampler(samples_per_pixel), max_depth)
    # to_gpu returns (gpu_scene, preserve) - preserve keeps GPU arrays alive during kernel execution
    gpu_scene, _preserve = Hikari.to_gpu(ArrayType, scene)
    gpu_film = Hikari.to_gpu(ArrayType, film)
    integrator(gpu_scene, gpu_film, camera[])
    # Copy result from GPU film back to CPU
    return Array(gpu_film.framebuffer)
end


function render_interactive(mscene::Makie.Scene; backend, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    imsub = Scene(mscene)
    imgp = image!(imsub, -1 .. 1, -1 .. 1, film.framebuffer, uv_transform=(:rotr90, :flip_y))
    integrator = Hikari.WhittedIntegrator(Hikari.UniformSampler(1), max_depth)
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
        @time integrator(scene, film, camera[])
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

# Export TraceMakie-specific types
export Screen, ScreenConfig, Whitted, activate!, colorbuffer

# re-export Makie, including deprecated names
for name in names(Makie, all=true)
    if Base.isexported(Makie, name)
        @eval using Makie: $(name)
        @eval export $(name)
    end
end

end
