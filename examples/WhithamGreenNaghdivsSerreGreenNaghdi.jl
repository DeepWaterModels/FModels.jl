# #
# Reproduces the figures in the Appendix of the HDR of V. Duchêne
# comparing solutions to the water waves system with predictions of
# the Serre-Green-Naghdi and Whitham-Green-Naghdi systems
# #
@info "Defines function Integrate"

using ShallowWaterModels,FFTW,Plots,LinearAlgebra,ProgressMeter;
include("../src/models/WhithamGreenNaghdi.jl")
include("../src/models/WaterWaves.jl")
include("../src/Figures.jl")
pyplot()
#include("../src/LoadSave.jl")
#using JLD


#----
"""
	`Integrate(scenario;kwargs)

Integrates in time the water waves system as well as SGN and WGN
with an initial datum being a heap of water with zero velocity

Other arguments are optional:
- `μ` is the shallowness parameter,
- `ϵ` in the nonlinearity parameter,
- `p` is the power in the initial datum `η = exp(-|x|^p)`
- `N` the number of collocation points,
- `L` the half-length of the mesh,
- `T` the final time of integration,
- `dt` the timestep,
- `dealias`: dealiasing with Orlicz rule `1-dealias/(dealias+2)` (default is `0`, i.e. no dealiasing),
- `iterate`: use GMRES if `true`, and LU decomposition otherwise (default is `true`),
- `precond` gives some choice in the preconditioner for GMRES,
- `name`: a string used to save the figures.

Return `(problems,plt)` where `problems` contains all the information and `plt` a plot of the final time solution.
"""
function Integrate(;μ=.1,ϵ=.1,p=2,N=2^10,L=15,T=10,dt=0.001,dealias=0,iterate=true,precond=true,name=nothing)

	if name != nothing ns=floor(Int,max(1,T/dt/100)) else ns=1 end
	param = ( μ  = μ, ϵ  = ϵ,
				N  = N, L  = L,
	            T  = T, dt = dt,
				ns=ns )

	mesh=Mesh(param)

	#---- preparation

	k=mesh.k
	if precond > 0
		precond = Diagonal( 1 .+ μ/3*(precond^2*k).^2 )
	elseif precond < 0
		precond = Diagonal( (1 .+ μ/3000*k.^8)  )
	end
	η = exp.(-abs.(mesh.x ).^p)
	u = 0*η
	v = 0*η
	init     = Init(mesh,η,v)

	modelWW	 = WaterWaves(param;dealias=dealias)
	modelSGN = WhithamGreenNaghdi(param;SGN=true, ktol=0, gtol=1e-12, iterate=iterate, precond = precond, dealias = dealias)
	modelWGN = WhithamGreenNaghdi(param;SGN=false, ktol=0, gtol=1e-12, iterate=iterate, precond = precond, dealias = dealias)

	problems = []
	push!(problems, Problem(modelWW, init, param) )
	push!(problems, Problem(modelSGN, init, param) )
	push!(problems, Problem(modelWGN, init, param) )


	#---- computation
	for problem in problems
		solve!( problem )
	end

	#---- analysis

	(ηfin,vfin,ufin)   =  modelSGN.mapfrofull(last(problems[2].data.U))
	E(η,u,v) = sum(η.^2 .+ (1 .+ param.ϵ*η).*u.*v)
	dE(η1,u1,v1,η2,u2,v2) = sum(η1.^2-η2.^2) + sum((1 .+ param.ϵ*η1).*u1.*v1 - (1 .+ param.ϵ*η2).*u2.*v2)

	print(string("Preserved energy with normalized precision ",dE(η,u,v,ηfin,ufin,vfin)/E(η,u,v),"\n"))

	(ηww,vww,xww,t)=solution(problems[1],t=1)
	(ηSGN,vSGN)=solution(problems[2],x=xww,t=1)
	(ηWGN,vWGN)=solution(problems[3],x=xww,t=1)

	DiffT1 = [norm(ηww-ηSGN,Inf) norm(ηww-ηWGN,Inf)]
	print(string("Error of SGN , WGN: ",round.(DiffT1, sigdigits=3),"\n"))

	(ηww,vww,xww,t)=solution(problems[1])
	(ηSGN,vSGN)=solution(problems[2],x=xww)
	(ηWGN,vWGN)=solution(problems[3],x=xww)

	Diff = [norm(ηww-ηSGN,Inf) norm(ηww-ηWGN,Inf)]
	print(string("Error of SGN , WGN: ",round.(Diff, sigdigits=3),"\n"))

	#---- plots

	plt = plot_solution(problems)
	display(plt)


	if name != nothing
		#save(problem,name);
		plt = plot_solution(problems;fourier=false)
		xlims!(plt,(first(xww),0))
		plot!(plt,legend=:topright)
		savefig(string(name,".pdf"));


		plot(xww, [ηww-ηSGN ηww-ηWGN],
			color=[:2 :3],primary=false)
		ind=range(1,step=length(xww)÷32,length=17);
		plot!(xww[ind], [ηww[ind]-ηSGN[ind] ηww[ind]-ηWGN[ind]],
			color=[:2 :3],marker=([:d :ut], 8, Plots.stroke(1, :gray)),linewidth=0,
			xlims=(first(xww),0),
		  title=string("difference (surface deformation) at t=",t),
		  label=["ww-SGN" "ww-WGN"])
		savefig(string(name,"-diff.pdf"));
		x = mesh.x
 		k = mesh.k

		# problem=problems[2];
		# ts = problem.times.ts
		# zs=zeros(param.N,length(ts));
		# @showprogress 1 for i in 1:length(ts)
		# 	zs[:,i].=real.(ifft(problem.data.U[i][:,1]))
		# end
		# plt3=plot()
		# my_cg = cgrad([:blue,:green])
		# surface!(plt3,x,ts[end:-1:1],zs',view_angle=(20,30), color = my_cg)
		# #display(plt3)
		# savefig(string(name,"-evol.pdf"));

		create_animation(problems;fourier=false,xlims=(first(xww),0),ylims=nothing,name=name)

	end
	display(plt)
	return problems,plt
end
nothing

#(problems,plt)=Integrate(p=3,μ=0.001,ϵ=0.125,name="mu0001eps0125");
