# # Workspace
#
#---- import libraries

using ShallowWaterModels,Plots,FFTW,Statistics;gr()
#include("../src/dependencies.jl")

include("../src/models/WaterWaves.jl")
include("../src/models/WhithamGreenNaghdi.jl")
include("../src/Figures.jl")


# if there is an error at this step, try commenting the lines above and commenting out the line below
#include("../src/dependencies.jl")

#---- parameters
param = ( μ  = .25,
			ϵ  = 0.1,
        	N  = 2^10, 	# number of collocation points
            L  = 15,	# size of the mesh (-L,L)
            T  = 10,		# final time of computation
            dt = 0.005  # timestep
				);
#---- initial data
ζ(x) = exp.(-abs.(x).^4);
v(x) = zero(x);
init = Init(ζ,v);
# ζ et v sont des fonctions.
# Init les mets sous une forme utilisable par le programme


#---- models to compare
models=[]
push!(models,WaterWaves(param;dealias=0))  # dealias = 1 pour dealiasing
push!(models,WhithamGreenNaghdi(param;SGN=true,dealias=0,iterate=true,precond=true)) # SGN=true => Green-Naghdi
push!(models,WhithamGreenNaghdi(param;SGN=false,dealias=0,iterate=true,precond=true))# SGN=false=> Whitham-Green-Naghdi
# ?WhithamGreenNaghdi ou ?WaterWaves pour voir les options disponibles

problems = []
for model in models
	push!(problems, Problem(model, init, param) )
end

#---- computation
for problem in problems
	@time solve!( problem )
end

#---- visualization
plt = plot_solution(problems;fourier=false)
savefig(plt,"plot.pdf") # ou "plot.png" par exemple

# pour visualiser autre chose, les différences par exemple
(η1,v1,x1,t)=solution(problems[1])
(η2,v2,x2,t)=solution(problems[2],x=x1)
(η3,v3,x3,t)=solution(problems[3],x=x1)

plt = plot(x1,η1-η2,label="diff 1")
plot!(plt,x1,η1-η3,label="diff 2")
plot!(xlims=(0,10))
plot!(ylims=(-0.1,0.1))


# pour faire un film
anim = create_animation(problems;fourier=false,ylims=(-0.5,1.1))
gif(anim, "SGNvsWGN.gif", fps=15)
#mov(anim, "SGNvsWGN.mov", fps=15)
#mp4(anim, "SGNvsWGN.mp4", fps=15)
