# Test allocations for li_iterative
# Run this after setting up a TraceMakie screen (e.g., from lego.jl)

using Hikari: li_iterative, UniformSampler, CameraSample, Point2f, generate_ray_differential
using BenchmarkTools

# Assumes `screen` is already defined from running the lego example
scene_obj = screen.state.hikari_scene
camera = screen.state.camera[]

# Create test inputs
sampler = UniformSampler(16)
max_depth = Int32(5)
cam_sample = CameraSample(Point2f(400, 300), Point2f(0.5, 0.5), 0f0)
ray, _ = generate_ray_differential(camera, cam_sample)

# Warm up
li_iterative(sampler, max_depth, ray, scene_obj)

# Test allocations
allocs = @allocated li_iterative(sampler, max_depth, ray, scene_obj)
println("Allocations: $allocs bytes")

# Benchmark
println("\nBenchmark:")
@btime li_iterative($sampler, $max_depth, $ray, $scene_obj)
