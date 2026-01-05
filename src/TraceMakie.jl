module TraceMakie

using Makie, Hikari, Colors, LinearAlgebra, GeometryBasics, Raycore, KernelAbstractions
using Makie: Observable, on, colorbuffer, to_value
using Makie: Quaternionf
using GeometryBasics: VecTypes
using Colors: N0f8
import Makie.Observables

# =============================================================================
# ScreenConfig
# =============================================================================

# Re-export integrators from Hikari for convenience
const Whitted = Hikari.Whitted
const SPPM = Hikari.SPPM
const FastWavefront = Hikari.FastWavefront

"""
    ScreenConfig

Configuration for TraceMakie rendering.

* `integrator`: The integrator to use for rendering (default: `Whitted()`)
  - `Whitted(; samples=8, max_depth=5)` - Fast Whitted-style ray tracing
  - `SPPM(; search_radius=0.075, max_depth=5, iterations=100)` - Stochastic progressive photon mapping
  - `FastWavefront(; samples=4)` - GPU-optimized wavefront path tracing
* `exposure`: Exposure multiplier for postprocessing (default: 1.0)
* `tonemap`: Tonemapping method (default: :aces)
  - `:reinhard` - Simple Reinhard L/(1+L)
  - `:reinhard_extended` - Extended Reinhard with white point
  - `:aces` - ACES filmic (industry standard)
  - `:uncharted2` - Uncharted 2 filmic
  - `:filmic` - Hejl-Dawson filmic
  - `nothing` - No tonemapping (linear clamp)
* `gamma`: Gamma correction value (default: 2.2, use `nothing` to skip)
* `backend`: Array type for rendering (default: `Array` for CPU)
  - `Array` - CPU rendering
  - `ROCArray` - AMD GPU via AMDGPU.jl
  - `CuArray` - NVIDIA GPU via CUDA.jl
"""
struct ScreenConfig
    integrator::Hikari.Integrator
    exposure::Float32
    tonemap::Union{Symbol, Nothing}
    gamma::Union{Float32, Nothing}
    backend::Type  # Array type: Array for CPU, ROCArray/CuArray for GPU

    function ScreenConfig(integrator, exposure, tonemap, gamma, backend=Array)
        actual_integrator = integrator isa Makie.Automatic ? Whitted() : integrator
        actual_exposure = Float32(exposure)
        actual_gamma = isnothing(gamma) ? nothing : Float32(gamma)
        return new(actual_integrator, actual_exposure, tonemap, actual_gamma, backend)
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
    PlotInfo(plot, handle, transform_obs, obs_funcs, count=1, per_inst=false, first_idx=0) = new(plot, handle, transform_obs, obs_funcs, count, per_inst, first_idx)
end


"""
    PlotUpdateInfo

Tracks a plot and its computed key for polling updates via the compute graph.
The update function is registered in the plot's attributes and updates
the corresponding Hikari material/geometry in-place when polled.
"""
struct PlotUpdateInfo
    plot::Makie.AbstractPlot
    computed_key::Symbol
end

"""
    TraceMakieState

Holds the state needed to synchronize a Makie scene with a Hikari ray tracing scene.
Supports dynamic updates to transformations via TLAS refit, and material/geometry
updates via the compute graph polling mechanism.
"""
mutable struct TraceMakieState
    plot_infos::Vector{PlotInfo}
    film::Hikari.Film  # Can be CPU (Array) or GPU (ROCArray/CuArray) - types differentiate
    camera::Observable
    needs_refit::Bool  # Flag to track if TLAS needs refit
    hikari_scene::Hikari.AbstractScene  # Contains the TLAS via hikari_scene.aggregate.accel (Scene for CPU, ImmutableScene for GPU)
    preserve::Vector{Any}  # Keep GPU arrays alive (empty for CPU)
    update_infos::Vector{PlotUpdateInfo}  # Track plots for compute graph polling
    needs_film_clear::Bool  # Flag to indicate data changed and film should be cleared
end

# Helper to get TLAS from state (it's inside hikari_scene.aggregate.accel)
get_tlas(state::TraceMakieState) = state.hikari_scene.aggregate.accel

"""
    register_plot_updates!(state::TraceMakieState, info::PlotInfo, material, material_idx)

Register a plot for in-place material/geometry updates using the compute graph.
This is called during scene conversion for each plot to set up reactive updates.

When polled before rendering, the compute graph evaluates any dirty attributes
and updates the corresponding Hikari structures in-place.
"""
function register_plot_updates!(state::TraceMakieState, info::PlotInfo, material, material_idx)
    # Dispatch to plot-specific registration
    _register_plot_updates!(state, info, material, material_idx)
end

# Default: no updates for unsupported plot types
function _register_plot_updates!(state::TraceMakieState, info::PlotInfo, material, material_idx)
    # No-op for plots without special update handling
end

# Volume plots: update density array in-place
function _register_plot_updates!(state::TraceMakieState, info::PlotInfo, cloud::Hikari.CloudVolume, material_idx)
    plot = info.plot
    plot isa Makie.Plot{Makie.volume} || return
    attr = plot.attributes
    computed_key = :tracemakie_volume_update

    # Skip if already registered (e.g., from previous colorbuffer call)
    if haskey(attr, computed_key)
        push!(state.update_infos, PlotUpdateInfo(plot, computed_key))
        return
    end

    # Register computation that watches the volume data
    # Must return a tuple with one element per output
    Makie.register_computation!(attr, [:volume], [computed_key]) do (vol_data,), changed, cached
        if changed.volume
            # Volume data changed - update CloudVolume density in-place
            if size(vol_data) == size(cloud.density)
                cloud.density .= Float32.(vol_data)
            else
                @warn "Volume size mismatch: $(size(vol_data)) vs $(size(cloud.density))"
            end
            state.needs_film_clear = true
            return (true,)
        end
        return isnothing(cached) ? (false,) : cached
    end

    push!(state.update_infos, PlotUpdateInfo(plot, computed_key))
end

# Generic material handler - dispatches based on plot type
function _register_plot_updates!(state::TraceMakieState, info::PlotInfo, mat::Hikari.Material, material_idx)
    plot = info.plot

    # MeshScatter: update positions, markersize, rotation -> refit TLAS
    if plot isa Makie.Plot{Makie.meshscatter} && info.instance_count > 1
        attr = plot.attributes
        computed_key = :tracemakie_meshscatter_update

        # Skip if already registered (e.g., from previous colorbuffer call)
        if haskey(attr, computed_key)
            push!(state.update_infos, PlotUpdateInfo(plot, computed_key))
            return
        end

        # Watch positions, markersize, and rotation
        # Must return a tuple with one element per output
        Makie.register_computation!(attr, [:positions, :markersize, :rotation], [computed_key]) do (positions, markersize, rotation), changed, cached
            if changed.positions || changed.markersize || changed.rotation
                state.needs_refit = true
                state.needs_film_clear = true
                return (true,)
            end
            return isnothing(cached) ? (false,) : cached
        end

        push!(state.update_infos, PlotUpdateInfo(plot, computed_key))

    # Mesh: update color in-place (for materials with mutable Texture)
    elseif plot isa Makie.Plot{Makie.mesh}
        # Try to get the texture from the material for in-place updates
        tex = _get_material_texture(mat)
        if !isnothing(tex) && tex isa Hikari.Texture
            attr = plot.attributes
            computed_key = :tracemakie_mesh_color_update

            # Skip if already registered
            if haskey(attr, computed_key)
                push!(state.update_infos, PlotUpdateInfo(plot, computed_key))
                return
            end

            # Must return a tuple with one element per output
            Makie.register_computation!(attr, [:color], [computed_key]) do (color,), changed, cached
                if changed.color
                    if color isa Colorant
                        tex.data = to_spectrum(to_color(color))
                    elseif color isa AbstractVector{<:Colorant}
                        tex.data = to_spectrum.(color)
                    elseif color isa AbstractMatrix{<:Colorant}
                        tex.data = to_spectrum.(color)
                    end
                    state.needs_film_clear = true
                    return (true,)
                end
                return isnothing(cached) ? (false,) : cached
            end

            push!(state.update_infos, PlotUpdateInfo(plot, computed_key))
        end
    end
end

# Helper to extract mutable texture from material (for in-place color updates)
_get_material_texture(mat::Hikari.MatteMaterial) = mat.Kd isa Hikari.Texture ? mat.Kd : nothing
_get_material_texture(mat::Hikari.PlasticMaterial) = mat.Kd isa Hikari.Texture ? mat.Kd : nothing
_get_material_texture(mat::Hikari.MetalMaterial) = mat.reflectance isa Hikari.Texture ? mat.reflectance : nothing
_get_material_texture(mat::Hikari.Material) = nothing

"""
    poll_updates!(state::TraceMakieState)

Poll all registered plots for updates. This triggers the compute graph
to evaluate any dirty attributes and update Hikari structures in-place.
Should be called before each render frame.

Returns true if any updates occurred (film should be cleared).
"""
function poll_updates!(state::TraceMakieState)
    had_updates = false
    for info in state.update_infos
        # Access the computed node to trigger resolution
        if haskey(info.plot.attributes, info.computed_key)
            computed = info.plot.attributes[info.computed_key]
            result = computed[]
            # Result is a tuple like (true,) or (false,)
            if result isa Tuple && length(result) >= 1 && result[1] === true
                had_updates = true
            end
        end
    end
    if state.needs_film_clear
        state.needs_film_clear = false
        return true
    end
    return had_updates
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

function Base.show(io::IO, screen::Screen)
    scene_str = isnothing(screen.scene) ? "nothing" : "Scene($(size(screen.scene)))"
    backend_name = nameof(screen.config.backend)
    integrator_name = nameof(typeof(screen.config.integrator))
    print(io, "Screen($scene_str, backend=$backend_name, integrator=$integrator_name)")
end

function Base.show(io::IO, ::MIME"text/plain", screen::Screen)
    println(io, "TraceMakie.Screen")
    if !isnothing(screen.scene)
        println(io, "  Scene size: ", size(screen.scene))
        if !isnothing(screen.state)
            println(io, "  Plots: ", length(screen.state.plot_infos))
        end
    else
        println(io, "  Scene: not attached")
    end
    println(io, "  Backend: ", nameof(screen.config.backend))
    println(io, "  Integrator: ", nameof(typeof(screen.config.integrator)))
    print(io, "  Exposure: ", screen.config.exposure)
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

function Makie.apply_screen_config!(screen::Screen, config::ScreenConfig, scene::Scene, args...)
    # Check if backend changed - if so, we need to recreate the screen entirely
    if screen.config.backend !== config.backend
        # Backend changed, need new screen with new state
        return Screen(scene, config)
    end

    # Check if integrator changed - if so, invalidate state to force re-render
    if typeof(screen.config.integrator) !== typeof(config.integrator)
        screen.state = nothing
    end

    # Update the config (postprocessing params like exposure/tonemap/gamma)
    screen.config = config
    return screen
end
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

    # Clear film and render (scene/film are already CPU or GPU based on backend)
    Hikari.clear!(state.film)
    camera = state.camera[]
    screen.config.integrator(state.hikari_scene, state.film, camera)

    return state.film
end

using ImageCore

function Makie.colorbuffer(screen::Screen, format::Makie.ImageStorageFormat = Makie.JuliaNative; figure = nothing)
    if isnothing(screen.state)
        display(screen, screen.scene; figure = figure)
    end

    render!(screen)

    # Apply postprocessing on GPU/CPU (tonemapping, gamma, exposure)
    config = screen.config
    Hikari.postprocess!(screen.state.film;
        exposure = config.exposure,
        tonemap = config.tonemap,
        gamma = config.gamma
    )

    # Copy postprocess buffer to CPU if on GPU, then convert to RGB{N0f8}
    result = Array(map(clamp01nan, screen.state.film.postprocess))

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

    # Apply postprocessing (works on GPU or CPU)
    Hikari.postprocess!(screen.state.film;
        exposure = exp_val,
        tonemap = tm_val,
        gamma = gamma_val
    )

    # Copy to CPU if on GPU, then convert to RGB{N0f8}
    postprocess_cpu = Array(screen.state.film.postprocess)
    result = map(postprocess_cpu) do c
        RGB{N0f8}(c.r, c.g, c.b)
    end

    return result
end

function Base.display(screen::Screen, scene::Scene; figure = nothing, display_kw...)
    screen.scene = scene
    screen.state = convert_scene_with_state(scene, screen.config.backend)
    return screen
end

function Base.insert!(screen::Screen, scene::Scene, plot::AbstractPlot)
    # For now, rebuild the entire state when plots change
    # Future: incremental updates
    if !isnothing(screen.state)
        screen.state = convert_scene_with_state(scene, screen.config.backend)
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
TraceMakie.activate!(integrator = TraceMakie.Whitted(samples=16, max_depth=8))

# Configure postprocessing
TraceMakie.activate!(exposure = 1.5, tonemap = :reinhard, gamma = 2.2)
```
"""
function activate!(; screen_config...)
    # Register TraceMakie's default theme if not already registered
    key = :TraceMakie
    Makie.set_screen_config!(TraceMakie, screen_config)
    Makie.set_active_backend!(TraceMakie)
    return
end

function __init__()
    # Register TraceMakie's default theme at init time (before activate)
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
    Raycore.update_transform!(get_tlas(state), info.handle, transform)
    state.needs_refit = true
end

"""
    refit_if_needed!(state::TraceMakieState)

Refit the TLAS if any transforms have changed.
"""
function refit_if_needed!(state::TraceMakieState)
    if state.needs_refit
        Raycore.refit_tlas!(get_tlas(state))
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

    # Create texture from color - updates are handled by compute graph registration
    tex = nothing
    if color isa AbstractMatrix{<:Number}
        # Use Makie's compute_colors to apply colormap
        computed = Makie.compute_colors(plot.attributes)
        tex = Hikari.Texture(to_spectrum(computed))
    elseif color isa AbstractMatrix{<:Colorant}
        tex = Hikari.Texture(to_spectrum(color))
    elseif color isa AbstractVector{<:Colorant}
        # Per-instance colors (e.g., for meshscatter)
        tex = Hikari.Texture(to_spectrum.(color))
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
    r = Tesselation(Rect2f((0, 0), (1, 1)), size(z[]))
    faces = decompose(GLTriangleFace, r)
    uv = decompose_uv(r)
    mesh = normal_mesh(GeometryBasics.Mesh(vec(positions[]), faces, uv=uv))

    # Convert to TriangleMesh using Raycore
    tmesh = Raycore.TriangleMesh(mesh)

    # Extract material - Surface uses z values for colormapping by default
    # Use Makie's compute_colors to get the colormapped texture
    material = extract_surface_material(plot)
    return Hikari.GeometricPrimitive(tmesh, material)
end

"""
Extract material for Surface plots, using Makie's color computation system.
"""
function extract_surface_material(plot::Makie.Surface)
    # Check if material is explicitly provided
    has_material = haskey(plot, :material) && !isnothing(to_value(plot.material))
    material_template = has_material ? to_value(plot.material) : nothing

    # Get the color - Surface can have explicit color or use z values
    color = to_value(plot.color)

    if color isa AbstractMatrix{<:Colorant}
        # Explicit color matrix provided
        tex = Hikari.Texture(to_spectrum(color))
    elseif color isa Colorant || color isa Union{String, Symbol}
        # Single color for entire surface
        tex = Hikari.ConstantTexture(to_spectrum(to_color(color)))
    else
        # Use Makie's compute_colors to get colormapped texture from z values
        computed = Makie.compute_colors(plot.attributes)
        tex = Hikari.Texture(to_spectrum(computed))
    end

    if material_template isa Hikari.Material
        return merge_color_with_material(tex, material_template)
    else
        return Hikari.MatteMaterial(tex, Hikari.ConstantTexture(0.0f0))
    end
end

function to_trace_primitive(plot::Makie.Plot)
    return nothing
end

"""
Convert a Makie Volume plot to a CloudVolume material with bounding box mesh.

The volume data is converted to a CloudVolume which uses ray marching for
physically-based cloud/volume rendering with Henyey-Greenstein phase function.

CloudVolume parameters are passed via the `material` attribute as a NamedTuple:
```julia
volume(x, y, z, data; material=(;
    extinction_scale=10000f0,    # Controls optical density
    asymmetry_g=0.85f0,          # HG phase function asymmetry (0.85 for clouds)
    single_scatter_albedo=0.99f0 # Scattering vs absorption ratio
))
```
"""
function to_trace_primitive(plot::Makie.Volume)
    !plot.visible[] && return nothing

    # Get volume data from the .volume attribute
    vol_data = to_value(plot.volume)

    # Convert to Float32 density field
    density = Float32.(vol_data)

    # Get spatial extent from x, y, z attributes (EndPoints)
    x = to_value(plot.x)
    y = to_value(plot.y)
    z = to_value(plot.z)

    # EndPoints have .start and .stop, or can be indexed [1] and [2]
    x_min, x_max = x[1], x[2]
    y_min, y_max = y[1], y[2]
    z_min, z_max = z[1], z[2]

    origin = Point3f(x_min, y_min, z_min)
    extent = Vec3f(x_max - x_min, y_max - y_min, z_max - z_min)

    # Get CloudVolume parameters from material attribute (NamedTuple or Attributes)
    mat_params = haskey(plot, :material) ? to_value(plot.material) : nothing

    extinction_scale = 100.0f0
    asymmetry_g = 0.85f0
    single_scatter_albedo = 0.99f0

    if mat_params isa NamedTuple
        extinction_scale = Float32(get(mat_params, :extinction_scale, extinction_scale))
        asymmetry_g = Float32(get(mat_params, :asymmetry_g, asymmetry_g))
        single_scatter_albedo = Float32(get(mat_params, :single_scatter_albedo, single_scatter_albedo))
    elseif mat_params isa Makie.Attributes
        # Makie converts NamedTuple to Attributes - extract values
        if haskey(mat_params, :extinction_scale)
            extinction_scale = Float32(to_value(mat_params[:extinction_scale]))
        end
        if haskey(mat_params, :asymmetry_g)
            asymmetry_g = Float32(to_value(mat_params[:asymmetry_g]))
        end
        if haskey(mat_params, :single_scatter_albedo)
            single_scatter_albedo = Float32(to_value(mat_params[:single_scatter_albedo]))
        end
    end

    # Create CloudVolume material
    cloud = Hikari.CloudVolume(
        density;
        origin=origin,
        extent=extent,
        extinction_scale=extinction_scale,
        asymmetry_g=asymmetry_g,
        single_scatter_albedo=single_scatter_albedo
    )

    # Create bounding box mesh for the volume
    cloud_box_geo = Rect3f(origin, extent)
    cloud_box_mesh = Raycore.TriangleMesh(normal_mesh(cloud_box_geo))

    return (cloud_box_mesh, cloud)
end

function to_trace_primitive_with_transform(plot::Makie.Volume)
    prim = to_trace_primitive(plot)
    if isnothing(prim)
        return nothing
    end
    mesh, material = prim
    # Volume coordinates are already in world space, use identity transform
    return (mesh, material, Mat4f(LinearAlgebra.I))
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

function to_trace_light(light::Makie.SunSkyLight)
    # Convert Makie's SunSkyLight to Hikari's SunSkyLight
    # Hikari expects sun_intensity as an RGBSpectrum, scaled by the intensity multiplier
    sun_intensity = Hikari.RGBSpectrum(light.intensity)
    ground_albedo = Hikari.RGBSpectrum(light.ground_albedo.r, light.ground_albedo.g, light.ground_albedo.b)
    return Hikari.SunSkyLight(
        Vec3f(light.direction),
        sun_intensity;
        turbidity=light.turbidity,
        ground_albedo=ground_albedo,
        ground_enabled=light.ground_enabled,
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
    convert_scene_with_state(scene::Makie.Scene, backend::Type=Array) -> TraceMakieState

Convert a Makie scene to a TraceMakieState that supports dynamic transform updates.
Automatically watches plot transformations and syncs to TLAS.

The `backend` parameter specifies the array type:
- `Array` (default): CPU rendering
- `ROCArray`: AMD GPU rendering
- `CuArray`: NVIDIA GPU rendering
"""
function convert_scene_with_state(mscene::Makie.Scene, backend::Type=Array)
    resolution = Point2f(size(mscene))
    film = Hikari.Film(
        resolution;
        filter=Hikari.LanczosSincFilter(Point2f(1.0f0), 3.0f0),
        crop_bounds=Hikari.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        diagonal=1.0f0, scale=1.0f0,
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

    # Track plot-to-material mapping for registering updates later
    # Maps plot -> (material, material_index)
    plot_to_material = Dict{Makie.AbstractPlot, Tuple{Hikari.Material, Hikari.MaterialIndex}}()

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
                # Track material for MeshScatter (for compute graph updates)
                # Use first material for per-instance case, or the single material
                mat_for_update = has_per_instance_mats ? first(result.materials) : result.materials
                mat_idx_for_update = get_material_index(mat_for_update)
                plot_to_material[plot] = (mat_for_update, mat_idx_for_update)
            elseif result isa Vector
                # Multiple primitives from MetaMesh - each gets its own Instance
                first_idx = length(instances) + 1
                first_descriptor_idx = total_instance_descriptors + 1
                first_mat = nothing
                first_mat_idx = nothing
                for (mesh, mat, transform) in result
                    mat_index = get_material_index(mat)
                    push!(instances, Raycore.Instance(mesh, transform, mat_index))
                    if isnothing(first_mat)
                        first_mat = mat
                        first_mat_idx = mat_index
                    end
                end
                total_instance_descriptors += length(result)
                plot_to_instance_info[plot] = (first_idx, length(result), false, first_descriptor_idx)
                # Track first material for MetaMesh (for compute graph updates)
                if !isnothing(first_mat)
                    plot_to_material[plot] = (first_mat, first_mat_idx)
                end
            else
                mesh, mat, transform = result
                first_idx = length(instances) + 1
                first_descriptor_idx = total_instance_descriptors + 1
                mat_index = get_material_index(mat)
                push!(instances, Raycore.Instance(mesh, transform, mat_index))
                total_instance_descriptors += 1
                plot_to_instance_info[plot] = (first_idx, 1, false, first_descriptor_idx)
                # Track material for this plot (for compute graph updates)
                plot_to_material[plot] = (mat, mat_index)
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

    # Add ambient light if present, but skip if we already have SunSkyLight
    # (SunSkyLight provides its own ambient illumination from the sky)
    has_sunsky = any(l -> l isa Hikari.SunSkyLight, lights)
    if !has_sunsky && haskey(mscene.compute, :ambient_color)
        ambient_color = mscene.compute[:ambient_color][]
        if ambient_color != RGBf(0, 0, 0)
            push!(lights, Hikari.AmbientLight(to_spectrum(ambient_color)))
        end
    end

    if isempty(lights)
        error("Must have at least one light")
    end

    # Create hikari scene
    material_scene = Hikari.MaterialScene(tlas, materials)
    hikari_scene = Hikari.Scene(lights, material_scene)

    # Convert to GPU if backend is not Array
    preserve = Any[]
    if backend !== Array
        hikari_scene = Hikari.to_gpu(backend, hikari_scene)
        film = Hikari.to_gpu(backend, film)
    end

    state = TraceMakieState(plot_infos, film, camera, false, hikari_scene, preserve, PlotUpdateInfo[], false)

    # Register transform observers
    for info in plot_infos
        obs_func = on(info.transform_obs) do _
            update_plot_transform!(state, info)
        end
        push!(info.obs_funcs, obs_func)
    end

    # Register compute graph updates for each plot
    # Build a lookup from plot to PlotInfo
    plot_to_info = Dict{Makie.AbstractPlot, PlotInfo}(info.plot => info for info in plot_infos)
    for (plot, (mat, mat_idx)) in plot_to_material
        if haskey(plot_to_info, plot)
            register_plot_updates!(state, plot_to_info[plot], mat, mat_idx)
        end
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

Uses GPU-compatible index-based updates that work on both CPU and GPU TLAS.
"""
function sync_transforms!(state::TraceMakieState)
    tlas = get_tlas(state)
    # Get backend from film's framebuffer (works for both CPU and GPU)
    backend = KernelAbstractions.get_backend(state.film.framebuffer)

    for info in state.plot_infos
        if info.instance_count > 1 && info.plot isa Makie.MeshScatter
            # MeshScatter: update all instance transforms using batch kernel
            sync_meshscatter_transforms!(state, info, backend)
        else
            # Regular plot: single transform update using batch kernel with count=1
            transform = get_plot_transform(info.plot)
            transforms = KernelAbstractions.allocate(backend, Mat4f, 1)
            copyto!(transforms, [transform])
            Raycore.update_instance_transforms!(tlas, transforms, 1, info.first_instance_idx)
        end
    end
    Raycore.refit_tlas!(tlas; backend=backend)
    state.needs_refit = false
end

"""
    sync_meshscatter_transforms!(state::TraceMakieState, info::PlotInfo, backend)

Update all instance transforms for a MeshScatter plot.
Uses GPU-compatible batch kernel with offset support.
"""
function sync_meshscatter_transforms!(state::TraceMakieState, info::PlotInfo, backend)
    plot = info.plot
    positions = to_value(plot.positions)
    markersize = to_value(plot.markersize)
    rotation = to_value(plot.rotation)
    plot_transform = get_plot_transform(plot)

    # Compute transforms on CPU
    transforms_cpu = meshscatter_transforms(positions, markersize, rotation, plot_transform)

    # Convert to appropriate backend array type
    transforms = KernelAbstractions.allocate(backend, Mat4f, length(transforms_cpu))
    copyto!(transforms, transforms_cpu)

    # Use batch kernel with offset (works on CPU and GPU)
    tlas = get_tlas(state)
    Raycore.update_instance_transforms!(tlas, transforms, info.instance_count, info.first_instance_idx)
end

"""
    render_frame!(state::TraceMakieState; samples=1, max_depth=5) -> Matrix

Render a single frame using the current state. Syncs transforms and refits TLAS if needed.
"""
function render_frame!(state::TraceMakieState; samples=1, max_depth=5)
    refit_if_needed!(state)
    Hikari.clear!(state.film)
    integrator = Hikari.Whitted(samples=samples, max_depth=max_depth)
    integrator(state.hikari_scene, state.film, state.camera[])
    return state.film.framebuffer
end

function render_whitted(mscene::Makie.Scene; samples=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.Whitted(samples=samples, max_depth=max_depth)
    # Call integrator directly - it uses KernelAbstractions for CPU/GPU dispatch
    integrator(scene, film, camera[])
    return film.framebuffer
end

function render_sppm(mscene::Makie.Scene; search_radius=0.075f0, max_depth=5, iterations=100)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.SPPM(search_radius=search_radius, max_depth=max_depth, iterations=iterations)
    integrator(scene, film, camera[])
    return film.framebuffer
end

function render_gpu(mscene::Makie.Scene, ArrayType; samples=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Hikari.Whitted(samples=samples, max_depth=max_depth)
    # to_gpu uses Raycore.PRESERVE global to keep GPU arrays alive during kernel execution
    gpu_scene = Hikari.to_gpu(ArrayType, scene)
    gpu_film = Hikari.to_gpu(ArrayType, film)
    integrator(gpu_scene, gpu_film, camera[])
    # Copy result from GPU film back to CPU
    return Array(gpu_film.framebuffer)
end


"""
    render_interactive(mscene; backend, max_depth=5, exposure=1.0f0, tonemap=:aces, gamma=1.2f0, render_backend=Array)

Start an interactive ray-tracing render loop for a Makie scene.

The render loop continuously updates as the camera moves. Uses progressive rendering
with 1 sample per pixel per frame, accumulating samples over time for noise reduction.
When the camera moves or plot data changes, the film is cleared and accumulation restarts.

Plot data changes (volume data, material parameters, etc.) are detected via the compute graph
polling mechanism - no Observable callbacks needed.

Postprocessing parameters (exposure, tonemap, gamma) can be Observables for reactive updates.

# Arguments
- `mscene::Makie.Scene`: The Makie scene to render
- `backend`: The Makie backend to use for display (e.g., GLMakie)
- `max_depth=5`: Maximum ray bounces
- `exposure=1.0f0`: Exposure value (can be Observable)
- `tonemap=:aces`: Tonemapping method (can be Observable, options: :aces, :reinhard, :filmic, nothing)
- `gamma=1.2f0`: Gamma correction (can be Observable)
- `render_backend=Array`: Array type for rendering (Array for CPU, ROCArray/CuArray for GPU)

# Returns
A named tuple with handles for controlling the render:
- `stop`: Function to stop the render loop
"""
function render_interactive(mscene::Makie.Scene; max_depth=5,
                            exposure=1.0f0, tonemap=:aces, gamma=1.2f0, render_backend=Array)
    # Wrap non-Observable parameters in Observables for uniform handling
    exposure_obs = exposure isa Observable ? exposure : Observable(exposure)
    tonemap_obs = tonemap isa Observable ? tonemap : Observable(tonemap)
    gamma_obs = gamma isa Observable ? gamma : Observable(gamma)

    # Create integrator - always 1 spp for progressive rendering
    integrator = Hikari.Whitted(samples=1, max_depth=max_depth)

    # Create Screen with proper backend configuration
    config = ScreenConfig(integrator, Float32(exposure_obs[]), tonemap_obs[], gamma_obs[], render_backend)
    screen = Screen(nothing, nothing, config)

    # Initialize state via display
    display(screen, mscene)
    state = screen.state
    film = state.film
    camera = state.camera

    # Hide volume plots for GLMakie display (they'll be ray-traced)
    volume_plots = [p for p in mscene.plots if p isa Makie.Volume]
    for p in volume_plots
        p.visible[] = false
    end

    # Temporarily remove SunSkyLight which GLMakie doesn't support
    makie_lights = Makie.get_lights(mscene)
    sun_sky_lights = [l for l in makie_lights if l isa Makie.SunSkyLight]
    for l in sun_sky_lights
        filter!(x -> x !== l, makie_lights)
    end

    imsub = Scene(mscene)
    display_buffer = film.postprocess
    imgp = image!(imsub, -1 .. 1, -1 .. 1, Array(display_buffer), uv_transform=(:rotr90, :flip_y))

    # Restore SunSkyLight after display
    for l in sun_sky_lights
        push!(makie_lights, l)
    end

    # Restore volume visibility (they'll be hidden by the overlay image anyway)
    for p in volume_plots
        p.visible[] = true
    end

    cam_start = camera[]
    loki = Threads.ReentrantLock()
    cam_rendered = camera[]
    running = Ref(true)

    # Main render loop
    Base.errormonitor(Threads.@spawn while running[] && !Makie.isclosed(mscene)
        # Poll for plot data updates (volume data, material changes, etc.)
        # This triggers the compute graph to apply any pending in-place updates
        if poll_updates!(state)
            # Data changed - clear film to restart accumulation
            Hikari.clear!(film)
            lock(loki) do
                imgp.visible = false
            end
        end

        # Check camera change
        if cam_rendered != camera[]
            cam_rendered = camera[]
            Hikari.clear!(film)
            lock(loki) do
                imgp.visible = false
            end
        end

        # Render using screen's integrator
        @time screen.config.integrator(state.hikari_scene, film, camera[])

        # Apply postprocessing with current observable values
        current_tonemap = tonemap_obs[]
        tonemap_sym = current_tonemap isa Symbol ? current_tonemap : (isnothing(current_tonemap) ? nothing : Symbol(current_tonemap))
        Hikari.postprocess!(film; exposure=Float32(exposure_obs[]), tonemap=tonemap_sym, gamma=Float32(gamma_obs[]))

        lock(loki) do
            imgp[3] = Array(film.postprocess)
            imgp.visible = true
        end
        sleep(1/30)
    end)

    # Camera visibility thread
    Base.errormonitor(Threads.@spawn while running[] && !Makie.isclosed(mscene)
        lock(loki) do
            if cam_start != camera[]
                cam_start = camera[]
                imgp.visible = false
            end
        end
        sleep(1/30)
    end)

    # Return control handles
    return (
        stop = () -> (running[] = false),
        screen = screen,
    )
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
