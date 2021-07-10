# # Comparing several models for water waves
#
#md # [`notebook`](@__NBVIEWER_ROOT_URL__notebooks/WWvsXX.ipynb)
#

using Distributed
using ShallowWaterModels
include("../src/models/WaterWaves.jl")
include("../src/models/PseudoSpectral.jl")

rmprocs(workers())
addprocs(3)

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using ShallowWaterModels
    include("../src/models/WaterWaves.jl")
    include("../src/models/PseudoSpectral.jl")


end

using Test
using TimerOutputs

function run_simulation()

    reset_timer!()


    @everywhere begin


        #---- parameters
        param = ( μ  = .1,
       		  ϵ  = 1,
                  N  = 2^11,  # number of collocation points
                  L  = 10,    # size of the mesh (-L,L)
                  T  = 5,     # final time of computation
                  dt = 0.001, # timestep
        				);
        #---- initial data
        g(x) = exp.(-abs.(x).^4);
        z(x) = 0*exp.(-x.^2);
        init = Init(g,z);

        #---- models to compare
        models=[]
        push!(models,WaterWaves(param))
        push!(models,PseudoSpectral(param;order=2,dealias=1,lowpass=1/100))
        push!(models,PseudoSpectral(param;order=3,dealias=1,lowpass=1/100))


        function solve_problem(model)
            problem = Problem(model, init, param)
            ShallowWaterModels.solve!(problem)
            last(problem.data.U)
        end

    end

    @timeit "Consecutive solving" begin
        p2 = []
        for model in models
            println(model.label)
            push!(p2, solve_problem(model))
        end
    end

    @timeit "Distributed" begin

        rc = RemoteChannel(()->Channel(3));
        pids = collect(workers())
        @sync for model in models
             @async put!(rc, (model.label,solve_problem(model)))
        end

        results = (take!(rc) for _ in 1:3)

        p1 = Dict( k => v for (k,v) in results)

        @show keys(p1)
    end


    print_timer()

    #---- Tests
    @testset "Final" begin
     	@test p1["water waves"] == p2[1]
     	@test p1["WW2"] == p2[2]
     	@test p1["WW3"] == p2[3]
    end

end

run_simulation()
