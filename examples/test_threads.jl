# # Comparing several models for water waves
#
#md # [`notebook`](@__NBVIEWER_ROOT_URL__notebooks/WWvsXX.ipynb)
#
using ShallowWaterModels;
include("../src/models/WaterWaves.jl")
include("../src/models/PseudoSpectral.jl")
using Test
using TimerOutputs
import Base.Threads: @threads, @sync, @async, @spawn, nthreads, threadid

function run_models()

    reset_timer!()
    #---- parameters
    param = ( μ  = .1,
    			ϵ  = 1,
            	N  = 2^11, 	# number of collocation points
                L  = 10,	# size of the mesh (-L,L)
                T  = 5,		# final time of computation
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

    #---- Initialize
    function Problems(models)
    	problems = []
    	for model in models
    		push!(problems, Problem(model, init, param) )
    	end
    	return problems
    end

    #---- Method 1
    @timeit "Method 1 with threads" begin
        problems=Problems(models)
        solve!(problems)
        p1=(last(problems[1].data.U), last(problems[2].data.U) ,last(problems[3].data.U))
    end

    #---- Method 2
    @timeit "Method 2 with threads" begin
        problems=Problems(models)
        @sync for problem in problems
        	@spawn solve!( problem )
        end
        p2=(last(problems[1].data.U), last(problems[2].data.U) ,last(problems[3].data.U))
    end

    #---- Method 3
    @timeit "Consecutive solving" begin
        problems=Problems(models)
        for problem in problems
        	solve!( problem )
        end
        p3=(last(problems[1].data.U), last(problems[2].data.U) ,last(problems[3].data.U))
    end

    #---- Tests
    @testset "Final" begin
    	@test p1==p2
    	@test p2==p3
    end

    print_timer()

end

run_models()
