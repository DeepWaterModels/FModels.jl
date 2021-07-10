# #
# Reproduces the figures in the work of C. Klein and V. Duchêne
# on the Serre-Green-Naghdi and Whitham-Green-Naghdi systems
# #
@info "Defines functions PlotSolitaryWaveWGN1,PlotSolitaryWaveWGN2,PlotSolitaryWaveWGN3,PlotJacobianWGN,IntegrateSolitaryWaveWGN,StabilitySolitaryWaveWGN,IntegrateWGN"

using ShallowWaterModels,FFTW,Plots,LinearAlgebra,ProgressMeter;
include("../src/models/WhithamGreenNaghdi.jl")
include("../src/initialdata/SolitaryWaveWhithamGreenNaghdi.jl")
include("../src/Figures.jl")
include("../src/LoadSave.jl")
using JLD

#---- Figures 1 and 2
"""
	`PlotSolitaryWaveWGN1(;kwargs)`

First method to compute the WGN solitary wave.
Use GMRES-iterative method for the inversion of the Jacobian.

Arguments are all optional:
- `c` the velocity,
- `N` the number of collocation points,
- `L` the half-length of the mesh,
- `name`: a string used to save the figure as a name.pdf.

Return `(η,u,v,mesh)`, where `mesh.x` are the collocation points.
"""
function PlotSolitaryWaveWGN1(;c=2,N=2^10,L=10*π,name=nothing)
	param = ( μ  = 1,
			ϵ  = 1,
        	N  = N,
            L  = L,
			c  = c
				)

	(η,u,v,mesh) = SolitaryWaveWhithamGreenNaghdi(
				param; SGN = false,
				method=2, α = 0,
				tol =  1e-14, max_iter=10,
				ktol =0*1e-11, gtol = 1e-16,
				iterative = true, q=1,
				verbose = false)

	(ηGN,uGN) = SolitaryWaveWhithamGreenNaghdi(
				param; SGN = true, max_iter=0)

	plt = plot(layout=(1,2))
	plot!(plt[1,1], mesh.x, [u uGN];
	  title=string("c=",c),
	  xlabel = "x",
	  ylabel = "u",
	  label=["WGN" "SGN"])
	plot!(plt[1,2], fftshift(mesh.k),
	  [log10.(abs.(fftshift(fft(u)))) log10.(abs.(fftshift(fft(uGN))))];
	  title="frequency",
	  label=["WGN" "SGN"])
	display(plt)
	if name != nothing savefig(string(name,".pdf")); end
	return (η,u,v,mesh)
end


#---- Figure 3
"""
	PlotSolitaryWaveWGN2(;kwargs)

Second method to compute the WGN solitary wave. Same as `PlotSolitaryWaveWGN1`, but
use non-iterative method for the inversion of the Jacobian.
To be used with higher values of the velocity (default is `c=20`).
"""
function PlotSolitaryWaveWGN2(;c=20,L=10*π,N=2^10,name=nothing)
	param = ( μ  = 1,
		ϵ  = 1,
      	N  = N,
        L  = L,
		c = c )

	(η,u,v,mesh) = SolitaryWaveWhithamGreenNaghdi(
				param; SGN = false,
				method=2, α = 1, #ici α = 1 évite des oscillations important si c = 3 ou c = 20
				tol =  1e-14, max_iter=15,
				iterative = false, q=1,
				verbose = true)

	(ηGN,uGN) = SolitaryWaveWhithamGreenNaghdi(
				param; SGN = true, max_iter=0)

	plt = plot(layout=(1,2))
	plot!(plt[1,1], mesh.x, [u uGN];
	  title=string("c=",c),
	  xlabel = "u",
	  ylabel = "u/c",
	  label=["WGN" "SGN"])
	plot!(plt[1,2], fftshift(mesh.k),
	  [log10.(abs.(fftshift(fft(u)))) log10.(abs.(fftshift(fft(uGN))))];
	  title="frequency",
	  label=["WGN" "SGN"])
    display(plt)
	if name != nothing savefig(string(name,".pdf")); end
	return (η,u,v,mesh)
end
#---- Figure 4
"""
	PlotSolitaryWaveWGN3(;kwargs)

Third method to compute the WGN solitary wave. Same as `PlotSolitaryWaveWGN1`, but
- use non-iterative method for the inversion of the Jacobian; and
- use a rescaled equation.
To be used with highest values of the velocity (default is `c=100`).
"""
function PlotSolitaryWaveWGN3(;c=100,L=10*π,N=2^10,name=nothing)
	param = ( μ  = 1,
		ϵ  = 1,
      	N  = N,
        L  = L,
		c = c )

	(η,u,v,mesh) = SolitaryWaveWhithamGreenNaghdi(
				param; SGN = false,
				method=3, α = 1,
				tol =  1e-10, max_iter=10,
				iterative = false, q=1,
				verbose = true)
	(ηGN,uGN) = SolitaryWaveWhithamGreenNaghdi(
				param; SGN = true, max_iter=0)

	plt = plot(layout=(1,2))
	plot!(plt[1,1], mesh.x, [u/c uGN/c];
	  title=string("c=",c),
	  xlabel = "x",
	  ylabel = "u/c",
	  label=["WGN" "SGN"])
	plot!(plt[1,2], fftshift(mesh.k),
	  [log10.(abs.(fftshift(fft(u/c)))) log10.(abs.(fftshift(fft(uGN/c))))];
	  title="frequency",
	  label="WGN")
	display(plt)
	if name != nothing savefig(string(name,".pdf")); end
	return (η,u,v,mesh)
end

#------ Figure 6
"""
	`PlotJacobianWGN(;kwargs)`

Computes the Jacobian matrix to be inverted in SGN or WGN equation.

Arguments are all optional:
- `c` the velocity,
- `N` the number of collocation points,
- `L` the half-length of the mesh,
- `SGN`: use SGN if `true`, and WGN if `false` (default is `false`),
- `name`: a string used to save the figure as a name.pdf.

"""
function PlotJacobianWGN(;c=20,L=10*π,N=2^10,SGN=false,name=nothing)
	ϵ,μ,α=1,1,0

	(η,u,v,mesh) = SolitaryWaveWhithamGreenNaghdi(
				(c=c,L=L,N=N,μ=μ,ϵ=ϵ); SGN = SGN,
				method=2, α = 1, #ici α = 1 évite des oscillations important si c = 3 ou c = 20
				tol =  1e-12, max_iter=15,
				iterative = false, q=1,
				verbose = true)

	k,x=mesh.k,mesh.x
	if SGN == true
		F₀ = sqrt(μ)*1im * k
	else
		F₁ = tanh.(sqrt(μ)*abs.(k))./(sqrt(μ)*abs.(k))
		F₁[1] = 1
		F₀ = 1im * sqrt.(3*(1 ./F₁ .- 1)).*sign.(k)
	end
	x₀ = mesh.x[1]
	FFT = exp.(-1im*k*(x.-x₀)');
	IFFT = exp.(1im*k*(x.-x₀)')/length(x);
	M₀ = IFFT * Diagonal( F₀ )* FFT
	M(v) = Diagonal( v )


	dxu = real.(ifft(1im*k.*fft(u)))
	dxu ./= norm(dxu,2)
	hu = c ./(c .- ϵ*u)
	Fu = hu.* real.(ifft(F₀.*fft(u)))
	F2u = real.(ifft(F₀.*fft(hu.^2 .* Fu ) ))
	Du = c ./hu .+ 2*ϵ/3 * F2u ./ hu .+ ϵ^2/c * hu .* Fu.^2 .- (hu.^2)/c
	Jac = (-1/3 *M(1 ./ hu.^2 )* M₀ * M(hu.^3)* (c*M₀ .+ 3*ϵ * M( Fu ))
					.+ ϵ * M( hu .* Fu ) * M₀
					.+ M( Du ) .+ α*M( hu.^2 )*dxu*dxu' *M(1 ./ hu.^2 ) )
	Jacstar = -1/3 *M(1 ./ hu.^2 )* M₀ * M(hu.^3)* c*M₀

	plt = plot(layout=(1,2))
	surface!(plt[1,1],fftshift(k),fftshift(k)[N:-1:1],
		log10.(abs.(FFT*Jac*IFFT)),
		title = "Jacobian")
	surface!(plt[1,2],fftshift(k),fftshift(k)[N:-1:1],
		log10.(abs.(FFT*Jacstar*IFFT)),
		title = "non-diagonal part")
	display(plt)

	if name != nothing savefig(string(name,".pdf")); end
	return (Jac,Jacstar,FFT,IFFT)
end

#---- Figures 7 and 8
"""
	`IntegrateSolitaryWaveWGN(;kwargs)`

Integrates in time the SGN or WGN solitary wave.

Arguments are all optional:
- `c` the velocity,
- `N` the number of collocation points,
- `L` the half-length of the mesh,
- `T` the final time of integration,
- `dt` the timestep,
- `SGN`: use SGN if `true`, and WGN if `false` (default is `false`),
- `name`: a string used to save raw data and the figure.

Return `problem` of type `Problem`, containing all the information.
"""
function IntegrateSolitaryWaveWGN(;SGN=false,c=2,N=2^10,L=10*π,T=1,dt=1/2000,name=nothing)
	if name != nothing ns=floor(Int,max(1,T/dt/100)) else ns=1 end

	param = ( μ  = 1, ϵ  = 1, c = c,
				N  = N,
	            L  = L,
	            T  = T,
	            dt = dt, ns=ns)
	if SGN == true
		(ηinit,uinit,vinit,mesh) = SolitaryWaveWhithamGreenNaghdi(
				param; SGN = true, max_iter=0)
		(ηfin,ufin,vfin) = SolitaryWaveWhithamGreenNaghdi(
				param; x₀ = c*T, SGN = true, max_iter=0)
	else
		(ηinit,uinit,vinit,mesh) = SolitaryWaveWhithamGreenNaghdi(
				param; method = 3, SGN = false, max_iter=10,α=1,verbose=true, tol=1e-12)
	  	(ηfin,ufin,vfin) = SolitaryWaveWhithamGreenNaghdi(
				param; x₀ = c*T, method = 3, SGN = false, max_iter=10,α=1,verbose=true, tol=1e-12)
	end
	init     = Init(mesh,ηinit,vinit)
	model = WhithamGreenNaghdi(param;SGN=SGN, ktol=0, iterate=true, precond = true)
	problem = Problem(model, init, param)
	solve!( problem )

	(ηcomp,vcomp,ucomp)   =  model.mapfrofull(last(problem.data.U))

	E(η,u,v) = sum(η.^2 .+ (1 .+ param.ϵ*η).*u.*v)
	dE(η1,u1,v1,η2,u2,v2) = sum(η1.^2-η2.^2) + sum((1 .+ param.ϵ*η1).*u1.*v1 - (1 .+ param.ϵ*η2).*u2.*v2)

	print(string("absolute energy difference: ",dE(ηinit,uinit,vinit,ηcomp,ucomp,vcomp),"\n"))
	print(string("relative energy difference: ",dE(ηinit,uinit,vinit,ηcomp,ucomp,vcomp)/E(ηinit,uinit,vinit),"\n"))

	plt = plot(layout=(1,2))
	plot!(plt[1,1],fftshift(mesh.k),fftshift(log10.(abs.(fft(uinit))));
			title = "frequency",
			label = "initial",
			xlabel="x",
			ylabel="u")
	plot!(plt[1,1],fftshift(mesh.k),fftshift(log10.(abs.(fft(ucomp))));
			label = "final")
	plot!(plt[1,2],mesh.x,ufin-ucomp;
			title = string("error at time t=",problem.times.tfin),
			label="difference",
			xlabel="x",
			ylabel="δu")
	display(plt)

	if name != nothing
		savefig(string(name,".pdf"));
		create_animation(problem;name=string(name,"-anim.pdf"))
		plot_solution(problem)
		savefig(string(name,"-final.pdf"));
		save(problem,name);
	end
	return problem
end

#------ Figures 9 to 15

"""
	`StabilitySolitaryWaveWGN(;kwargs)`

Integrates in time a pertubed SGN or WGN solitary wave.

Arguments are all optional:
- `p` (1,2, or a real) is the type of perturbation:
	1. If `p=1`, then `u` is multiplied by `λ = 0.99`
	2. If `p=2`, then `u` is multiplied by `λ = 1.01`
	3. Otherwise, one adds `p exp(-x^2)` to `u`
- `c` the velocity,
- `N` the number of collocation points,
- `L` the half-length of the mesh,
- `T` the final time of integration,
- `dt` the timestep,
- `SGN`: use SGN if `true`, and WGN if `false` (default is `false`),
- `iterate`: use GMRES if `true`, and LU decomposition otherwise (default is `true`),
- `precond` gives some choice in the preconditioner for GMRES,
- `dealias` use 2/3 dealiasing rule if `1`, (default is `0`, i.e. no dealiasing),
- `name`: a string used to save raw data and the figures.

Return `problem` of type `Problem`, containing all the information.
"""
function StabilitySolitaryWaveWGN(;p=2,c=2,N=2^10,L=10*π,T=10,dt=10/10^4,SGN=false,precond=true,iterate=true,dealias=0,name=nothing)
	if name != nothing ns=floor(Int,max(1,T/dt/100)) else ns=1 end
	if p == 1
		λ = 0.99
	elseif p == 2
		λ = 1.01
	else
		λ = p
	end

	param = ( μ = 1, ϵ = 1, c = c, λ = λ,
				N  = N,
	            L  = L,
	            T  = T,
	            dt = dt, ns=ns,
				dealias = dealias,
				SGN=SGN,
				ktol=0*1e-6,
				gtol=1e-12,
				iterate=iterate)

	if SGN == true
		(η,u,v,mesh) = SolitaryWaveWhithamGreenNaghdi(
				param; SGN = true, max_iter=0)
	else
		(η,u,v,mesh) = SolitaryWaveWhithamGreenNaghdi(
				param; method = 3, tol=1e-14, SGN = false, max_iter=10,α=1,verbose=true)
	end
	k = mesh.k

	if precond > 0
		precond = Diagonal( 1 .+ μ/3*(precond^2*k).^2 )
	elseif precond < 0
		precond = Diagonal( (1 .+ μ/3000*k.^8)  )
	end
	if p == 1 || p == 2
		u= λ*u
	else
		u .+= λ*exp.(-mesh.x.^2)
	end
	if SGN == true
		F₀ = sqrt(param.μ)*1im * k
	else
		F₁ = tanh.(sqrt(param.μ)*abs.(k))./(sqrt(param.μ)*abs.(k))
		F₁[1] = 1
		F₀ = 1im * sqrt.(3*(1 ./F₁ .- 1)).*sign.(k)
	end
	DxF(v) = real.(ifft(F₀ .* fft(v)))
	h = 1 .+ param.ϵ*η
	v = u - 1/3 ./h .* (DxF(h.^3 .*DxF(u)))

	init = Init(mesh,η,v)
	model = WhithamGreenNaghdi(param;SGN=param.SGN, dealias = param.dealias, ktol=param.ktol, gtol=param.gtol, iterate=param.iterate, precond = precond)
	problem = Problem(model, init, param)
	solve!( problem )

	if name != nothing save(problem,name); end

	(ηfin,vfin,ufin)   =  model.mapfrofull(last(problem.data.U))

	E(η,u,v) = sum(η.^2 .+ (1 .+ param.ϵ*η).*u.*v)
	dE(η1,u1,v1,η2,u2,v2) = sum(η1.^2-η2.^2) + sum((1 .+ param.ϵ*η1).*u1.*v1 - (1 .+ param.ϵ*η2).*u2.*v2)

	print(string("normalized error: ",dE(η,u,v,ηfin,ufin,vfin)/E(η,u,v),"\n"))

	plt = plot(layout=(1,2))
	plot!(plt[1,1],fftshift(mesh.k),fftshift(log10.(abs.(fft(η))));
			title = "frequency",
			label = "initial")
	plot!(plt[1,1],fftshift(mesh.k),fftshift(log10.(abs.(fft(ηfin))));
			label = "final")
	plot!(plt[1,2],mesh.x,[η ηfin];
			title = string("at time t=",problem.times.tfin),
			label=["zeta initial" "zeta final"])
	plot!(plt[1,2],mesh.x,[u ufin];
			label=["u initial" "u final"])
	display(plt)
	if name != nothing savefig(string(name,"-final.pdf")); end

	ts = problem.times.ts
	x = mesh.x
	k = mesh.k
	X=interpolate(mesh,real.(ifft(problem.data.U[1][:,1])))[1].x
	zs=zeros(length(X),length(ts));
	for i in 1:length(ts)
		zs[:,i].=interpolate(mesh,real.(ifft(problem.data.U[i][:,1])))[2]
	end

	plt = plot()
	scatter!(plt,ts,maximum(abs.(zs),dims=1)',
			title="maximum of surface deformation",
			label="",
			xlabel="time t")
	display(plt)


	if name != nothing
		savefig(string(name,"-znorm.pdf"));

		us=zeros(length(X),length(ts));
		@showprogress 1 "Computing v..." for i in 1:length(ts)
			us[:,i].=interpolate(mesh,model.mapfro(problem.data.U[i])[2])[2]
		end
		plt = plot()
		scatter!(plt,ts,maximum(abs.(us),dims=1)',
				title="maximum of velocity",
				label="",
				xlabel="time t")
		display(plt)
		savefig(string(name,"-vnorm.pdf"));


		plt=plot()
		my_cg = cgrad([:blue,:green])
		#surface!(plt,X,ts,zs',view_angle=(20,30), color = my_cg)
		display(plt)
		savefig(string(name,"-evol.pdf"));

		create_animation(problem;name=string(name,"-anim"))
	end
	return problem
end

#---- Figures 16 to end
"""
	`IntegrateWGN(scenario;kwargs)

Integrates in time SGN or WGN with an initial data depending on a given `scenario`
- If `scenario = 1`, produces a dispersive shock wave for a unidirectional wave constructed from the Saint-Venant model.
- If `scenario = 2`, produces a dispersive shock wave for a unidirectional wave constructed from the Camassa-Holm model.
- If `scenario = 3`, the initial data challenges the non-cavitation assumption.

Other arguments are optional:
- `δ` is either
	- the square root of the shallowness parameter (the nonlinear parameter `ϵ=1`) if `scenario=1` or `2`;
	- the minimal depth if `scenario=3`
- `N` the number of collocation points,
- `L` the half-length of the mesh,
- `T` the final time of integration,
- `dt` the timestep,
- `SGN`: use SGN if `true`, and WGN if `false` (default is `false`),
- `iterate`: use GMRES if `true`, and LU decomposition otherwise (default is `true`),
- `precond` gives some choice in the preconditioner for GMRES,
- `dealias`: dealiasing with Orlicz rule `1-dealias/(dealias+2)` (default is `0`, i.e. no dealiasing);
- `name`: a string used to save raw data and the figures.

Return `(problem,plt)` where `problem` contains all the information and `plt` a plot of the final time solution.
"""
function IntegrateWGN(scenario;δ=0.1,N=2^11,L=3*π,x₀=-3,T= 5,dt = 5/10^4,SGN=false,dealias=0,iterate=true,precond=true,name=nothing)

	if name != nothing ns=floor(Int,max(1,T/dt/100)) else ns=1 end
	if scenario == 1 || scenario == 2
		μ  = δ^2
	else
		μ = 1
	end
	param = ( μ  = μ, ϵ  = 1,
				N  = N, L  = L,
	            T  = T, dt = dt,
				ns=ns )

	mesh=Mesh(param)
	k=mesh.k
	if precond > 0
		precond = Diagonal( 1 .+ μ/3*(precond^2*k).^2 )
	elseif precond < 0
		precond = Diagonal( (1 .+ μ/3000*k.^8)  )
	end


	if scenario == 1
		η= exp.(-(mesh.x .-x₀).^2)
		u= 2*sqrt.(1 .+ param.ϵ*η) .-2
	elseif scenario == 2
		function krasny!(v)
			v[abs.(v).<1e-14].=0
			return v
		end
		Dx(v) = real.(ifft( 1im*mesh.k.* krasny!(fft(v))))
		Dx2(v) = Dx(Dx(v))
		ϵ = param.ϵ;μ=param.μ;
		w = - (mesh.x.-x₀).* exp.(-(mesh.x.-x₀).^2)
		u = w .+ μ/12 *Dx2(w) .+ μ*ϵ/6* w.*Dx2(w)
		η = u .+ ϵ/4*u.^2 .- μ/6* Dx2( u+3*ϵ/4* u.^2) .- μ*ϵ/6 * u .* Dx2(u) .- 5*μ*ϵ/48 * Dx(u).^2
	elseif scenario == 3
		η = -(1-δ)*exp.(-mesh.x.^2)
		u = -mesh.x.*exp.(-mesh.x.^2)
	else
		error("the first argument must be 1, 2 or 3")
	end
	if SGN == true
		F₀ = sqrt(param.μ)*1im*mesh.k
	else
		F₁ 	= tanh.(sqrt(param.μ)*abs.(mesh.k))./(sqrt(param.μ)*abs.(mesh.k))
		F₁[1]= 1                 # Differentiation
		F₀   = 1im * sqrt.(3*(1 ./F₁ .- 1)).*sign.(mesh.k)
	end
	DxF(v) = real.(ifft(F₀ .* fft(v)))
	h=1 .+ param.ϵ*η
	v= u - 1/3 ./h .* (DxF(h.^3 .*DxF(u)))

	init     = Init(mesh,η,v)
	model = WhithamGreenNaghdi(param;SGN=SGN, ktol=0, gtol=1e-12, iterate=iterate, precond = precond, dealias = dealias)
	problem = Problem(model, init, param)
	solve!( problem )

	(ηfin,vfin,ufin)   =  model.mapfrofull(last(problem.data.U))
	E(η,u,v) = sum(η.^2 .+ (1 .+ param.ϵ*η).*u.*v)
	dE(η1,u1,v1,η2,u2,v2) = sum(η1.^2-η2.^2) + sum((1 .+ param.ϵ*η1).*u1.*v1 - (1 .+ param.ϵ*η2).*u2.*v2)

	print(string("normalized error: ",dE(η,u,v,ηfin,ufin,vfin)/E(η,u,v),"\n"))

	fftηfin=last(problem.data.U)[:,1]

	plt = plot(layout=(1,2))
	plot!(plt[1,1],fftshift(mesh.k),fftshift(log10.(abs.(fftηfin)));
			title = "frequency",label="")
	plot!(plt[1,2],mesh.x,real.(ifft(fftηfin));
			title = string("surface deformation at time t=",problem.times.tfin),
			label="")
	display(plt)

	if name != nothing
		savefig(string(name,".pdf"));
		save(problem,name);
		ts = problem.times.ts
		x = mesh.x
		k = mesh.k
		zs=zeros(param.N,length(ts));
		@showprogress 1 for i in 1:length(ts)
			zs[:,i].=real.(ifft(problem.data.U[i][:,1]))
		end
		plt3=plot()
		my_cg = cgrad([:blue,:green])
		surface!(plt3,x,ts,zs',view_angle=(20,30), color = my_cg)
		display(plt3)
		savefig(string(name,"-evol.pdf"));
		create_animation(problem;ylims=false,name=string(name,"-anim"))

	end
	display(plt)
	return problem,plt
end
nothing
