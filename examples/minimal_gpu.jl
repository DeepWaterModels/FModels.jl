using ProgressMeter
using FFTW, LinearAlgebra
using Test

include("../src/dependencies.jl")
include("../src/models/WhithamGreenNaghdiGPU.jl")

function run_cpu( param )

    mesh = Mesh(param) # construct mesh of collocation points, Fourier modes, etc.
    η = exp.(-(mesh.x) .^ 2)
    v = zero(η)  # Initial data
    init = Init(mesh, η, v)
    times = Times(param.dt, param.T)
    model = WhithamGreenNaghdi(param; iterate=true)
    solver = RK4(param;model= model)
    data  = Data(mapto(model, init))
    U = copy(last(data.U))
    dt = times.dt
    problem = Problem(model, init, param)
    solve!( problem )
    real(problem.data.U)

end

function run_gpu( param )

    mesh = Mesh(param) # construct mesh of collocation points, Fourier modes, etc.
    η = exp.(-(mesh.x) .^ 2)
    v = zero(η)  # Initial data
    init = Init(mesh, η, v)
    times = Times(param.dt, param.T)
    model = WhithamGreenNaghdiGPU(param)
    solver = RK4(param;model= model)
    data  = Data(mapto(model, init))
    U = copy(last(data.U))
    dt = times.dt
    problem = Problem(model, init, param)
    solve!( problem )
    real(problem.data.U)

end

param = (
        μ = 1,
        ϵ = 1,
        N = 2^11,       # number of collocation points
        L = 10π,        # mesh of size 2L
        T = 1.0,        # final time
        dt = 0.001,    # timestep
        ns = 1,         # stores data every ns time steps
      )

# trigger compilation
@time U_cpu = run_cpu(param)
@time U_gpu = run_gpu(param)

@test U_gpu ≈ U_cpu
