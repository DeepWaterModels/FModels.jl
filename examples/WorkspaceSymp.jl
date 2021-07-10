# # Workspace
#
#md # [`notebook`](@__NBVIEWER_ROOT_URL__notebooks/two_problems.ipynb)
#
#using ShallowWaterModels,ProgressMeter,Plots,FFTW,Statistics;gr()
include("../src/dependencies.jl")

include("../src/models/PseudoSpectral.jl")
include("../src/models/Spectral.jl")
include("../src/models/WaterWaves.jl")

include("../src/solvers/Euler.jl")
include("../src/solvers/EulerSymp.jl")



# if there is an error at this step, try commenting the lines above and commenting out the line below
#include("../src/dependencies.jl")

#---- parameters
param = ( μ  = .25,
			ϵ  = .1,
        	N  = 2^8, 	# number of collocation points
            L  = π,	# size of the mesh (-L,L)
            T  = 50,		# final time of computation
            dt = 0.001, # timestep
			ns = 50,   	# data stored every ns step
				);
#---- initial data
g(x) = exp.(-abs.(x).^4);
z(x) = 0 .*(x .+1).*exp.(-x.^2);
init = Init(g,z);

#---- building problems
model=WaterWaves(param)
solver1=Euler(model)
solver2=EulerSymp(model,Niter=5,implicit=1)
solver3=EulerSymp(model,Niter=5,implicit=2)
solver=RK4(model)


problems = []
push!(problems, Problem(model, init, param, solver=solver ))
push!(problems, Problem(model, init, param, solver=solver1 ))
push!(problems, Problem(model, init, param, solver=solver2 ))
push!(problems, Problem(model, init, param, solver=solver3 ))


#---- computation
for problem in problems
	@time solve!( problem )
end
#solve!(problems[3])
#---- visualization
include("../src/Figures.jl")
#using Plots
#gr()
(η,v,x,t)=solution(problems[1])
(η2,v2,x2,t)=solution(problems[2])
(η3,v3,x3,t)=solution(problems[3])
(η4,v4,x4,t)=solution(problems[4])

using Plots;gr()
plot(x1,[η-η2])
plot(x1,[η-η2 η-η3])
