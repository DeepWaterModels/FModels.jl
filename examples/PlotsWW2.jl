# #
# Reproduces the figures in the work of V. Duchêne and B. Mélinand
# on the quadratic pseudo-spectral method (WW2)
# #
@info "Defines function IntegrateWW2"

using ShallowWaterModels,FFTW,Plots,LinearAlgebra,ProgressMeter;
include("../src/models/PseudoSpectral.jl")
include("../src/Figures.jl")
include("../src/LoadSave.jl")
using JLD

#---- Figures 1 to 3

"""
	`IntegrateWW2(scenario;kwargs)

Integrates in time the WW2 or WGN with an initial data depending on a given `scenario`
The scenario correspond to different plots in [DM]

Other arguments are optional:
- `μ` the shallowness parameter (default is `1`),
- `ϵ` the nonlinearity parameter (default is `0.1`),
- `L` the half-length of the mesh,
- `N` the number of collocation points,
- `T` the final time of integration,
- `dt` the timestep,
- `dealias`: dealiasing with Orlicz rule `1-dealias/(dealias+2)` (default is `1`, i.e. 2/3 rule, `0` means no dealiasing);
- `δ` the strength of the rectifier,
- `reg` the order of the rectifier (as a regularizing operator),
- `name`: a string used to save raw data and the figures.

Return `(problem,plt)` where `problem` contains all the information and `plt` a plot of the final time solution.
"""
function IntegrateWW2(scenario;μ=1,ϵ=0.1,L=20,N=2^10,T= 10,dt = 0.001,dealias=1,δ=0.001,reg=1,Tint=nothing,name=nothing)

	if name != nothing ns=floor(Int,max(1,T/dt/100)) else ns=1 end

	if scenario == 1
		μ  = 1; ϵ = 0.1; L = 20; N = 2^9; T = 10; dt = 0.01;
		dealias = 0 ; # no dealiasing
		δ = 0 ; reg = 1; # no rectifier

	elseif scenario == 2
		μ  = 1; ϵ = 0.1; L = 20; N = 2^11; T = 1.2; dt = 0.01;
		dealias = 0 ; # no dealiasing
		δ = 0 ; reg = 1; # no rectifier

	elseif scenario == 3
		μ  = 1; ϵ = 0.1; L = 20; N = 2^12; T = 10; dt = 0.01;
		dealias = 1 ; # dealiasing with 2/3 rule
		δ = 0 ; reg = 1; # no rectifier

	elseif scenario == 4
		μ  = 1; ϵ = 0.1; L = 20; N = 2^14; T = 1.3; dt = 0.01;
		dealias = 1 ; # dealiasing with 2/3 rule
		δ = 0 ; reg = 1; # no rectifier

	elseif scenario == 5
		μ  = 1; ϵ = 0.1; L = 20; N = 2^14; T = 10; dt = 0.01;
		dealias = 1 ; # dealiasing with 2/3 rule
		δ = 0.01 ; reg = 2; # rectifier of order 2
		Tint = 2; #to print an indermediate time

	elseif scenario == 6
		μ  = 1; ϵ = 0.1; L = 20; N = 2^14; T = 10; dt = 0.01;
		dealias = 1 ; # dealiasing with 2/3 rule
		δ = 0.002 ; reg = 1; # rectifier of order 2
		Tint = 2; #to print an indermediate time

	else
		error("the first argument must be between 1 and 6")

	end
	param = ( μ  = μ, ϵ  = ϵ,
				N  = N, L  = L,
	            T  = T, dt = dt,
				ns=ns )

	mesh=Mesh(param)

	η(x)=exp.(-x.^2);v(x)=zero(x);
	init     = Init(η,v)
	model = PseudoSpectral(param;order = 2, δ = δ, reg = reg, dealias = dealias)
	problem = Problem(model, init, param)
	solve!( problem )

	(ηfin,vfin)   =  model.mapfro(last(problem.data.U))

	x=mesh.x;k=mesh.k;
	Tmu = -1im*tanh.(sqrt(μ)*k)
	tmu = tanh.(sqrt(μ)*k)./k;tmu[1]=sqrt(μ);
	∂ₓ= 1im*k

	e(η,v) = 1/2* ( η.^2 .+ v.*ifft(tmu.*fft(v)) +ϵ*η.*(v.^2-ifft(Tmu.*fft(v)).^2) );
	print(string("normalized error: ",sum(e(ηfin,vfin)-e(η(x),v(x)))/sum(e(η(x),v(x))),"\n"))



	if Tint == nothing
		plt = plot(layout=(2,1))

		plot!(plt[2,1],fftshift(k),fftshift(abs.(fft(ηfin)));
			title = "Fourier coefficients (log scale)",label="",yscale = :log10)
		plot!(plt[1,1],x,ηfin;
			title = string("surface deformation at time t=",problem.times.tfin),
			label="")

	else
		plt = plot(layout=(2,1))

		plot!(plt[1,1],x,ηfin;
				label=string("T=",T) )

		plot!(plt[2,1],fftshift(k),fftshift(abs.(fft(ηfin))).+eps();
				title = "Fourier coefficients (log scale)",label="",yscale = :log10)

		(ηint,vint)   =  solution(problem,t=Tint)

		plot!(plt[1,1],x,ηint;
				title = "surface deformation" ,
				label=string("T=",Tint) )
		plot!(plt[2,1],fftshift(k),fftshift(abs.(fft(ηint))).+eps();
				title = "Fourier coefficients (log scale)",label="",yscale = :log10)

	end
	display(plt)

	if name != nothing
		savefig(string(name,".pdf"));
		save(problem,name);
		create_animation(problem;ylims=false,name=name)

	end
	display(plt)
	return problem,plt
end
#nothing

IntegrateWW2(6)
