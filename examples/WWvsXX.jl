# # Comparingseveral models for water waves
#
#md # [`notebook`](@__NBVIEWER_ROOT_URL__notebooks/WWvsXX.ipynb)
#
using ShallowWaterModels;
include("../src/models/WaterWaves.jl")
include("../src/models/PseudoSpectral.jl")

# if there is an error at this step, try commenting the lines above and commenting out the line below
#include("../src/dependencies.jl")

#---- parameters
param = ( μ  = .1,
			ϵ  = 1,
        	N  = 2^12, 	# number of collocation points
            L  = 10,	# size of the mesh (-L,L)
            T  = 5,		# final time of computation
            dt = 0.001, # timestep
			ns = 50,   	# data stored every ns step
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
problems = []
for model in models
	push!(problems, Problem(model, init, param) )
end

#---- computation
for problem in problems
	solve!( problem )
end

#---- visualization
include("../src/Figures.jl")
#using Plots
#gr()

plt = plot_solution(problems)
display(plt)
savefig("WWvsXX.pdf")
anim = create_animation(problems)
gif(anim, "WWvsXX.gif", fps=15)
