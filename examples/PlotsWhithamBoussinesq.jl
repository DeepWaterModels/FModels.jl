# #
# A proof of concept:
# 1) PlotSolitaryWaveWhithamBoussinesq() numerically computes the solitary wave of a Whitham-Boussinesq equation with prescribed velocity
# 2) IntegrateSolitaryWaveWhithamBoussinesq() integrates the equation with time
# #
@info "Define functions PlotSolitaryWaveWhithamBoussinesq(),IntegrateSolitaryWaveWhithamBoussinesq()"

using ShallowWaterModels,FFTW,Plots;
include("../src/models/WhithamBoussinesq.jl")
include("../src/initialdata/SolitaryWaveWhithamBoussinesq.jl")
include("../src/Figures.jl")

#----
"""
	`PlotSolitaryWaveWhithamBoussinesq(;kwargs)`

Constructs and plots a solitary wave of the Whitham-Boussinesq equation.

All arguments are optional.
- `c` the velocity of the wave,
- `α` defines the model used,
- `L` the half-length of the mesh,
- `N` the number of collocation points,
- `μ` the shallowness parameter,
- `ϵ` the nonlinearity parameter.

"""
function PlotSolitaryWaveWhithamBoussinesq(;c=1.05,α=1/2,L=20,N=2^9,μ=0.1,ϵ=0.1)
	param = ( μ  = μ, ϵ = ϵ,
			N = N, L = L,
			c = c, α = α)

	(η,v,mesh) = SolitaryWaveWhithamBoussinesq( param;
										α = 1, model  = α, iterative = false, max_iter=30)

	ηSGN = (c^2-1)/ϵ*sech.(sqrt(3*(c^2-1)/(c^2)/μ)/2*(mesh.x)).^2
	uSGN = c*η./(1 .+ ϵ*η)
	k = mesh.k
	F₀ = sqrt(param.μ)*1im * k
	DxF(v) = real.(ifft(F₀ .* fft(v)))
	h = 1 .+ param.ϵ*ηSGN
	vSGN = uSGN - 1/3 ./h .* (DxF(h.^3 .*DxF(uSGN)))


	plt = plot(layout=(1,2))
	plot!(plt[1,1], mesh.x, [η v];
     title=string("initial data"),
     label=["ζ" "v"])
	plot!(plt[1,1], mesh.x, [ηSGN vSGN];
      title=string("initial data"),
      label=["ζSGN" "vSGN"])


    plot!(plt[1,2], fftshift(mesh.k),
     [log10.(abs.(fftshift(fft(η)))) log10.(abs.(fftshift(fft(v))))];
     title="frequency",
	 label=["η" "u"])
	plot!(plt[1,2], fftshift(mesh.k),
      [log10.(abs.(fftshift(fft(ηSGN)))) log10.(abs.(fftshift(fft(vSGN))))];
      title="frequency",
 	 label=["η" "u"])

	 display(plt)
end

"""
	`IntegrateSolitaryWaveWhithamBoussinesq()`

Integrates in time a Whitham-Boussinesq equation with a solitary wave initial data
"""
function IntegrateSolitaryWaveWhithamBoussinesq()
	param = ( μ  = .1,
			ϵ  = .1,
        	N  = 2^9,
            L  = 20,
			c  = 1.05,
			T  = 40/1.05,
            dt = 0.001/1.05,
			ns = 200,
			α	= 1/2,
			)

	(η,v,mesh) = SolitaryWaveWhithamBoussinesq( param;
										α = 1, model  = 1/2, iterative = false, max_iter=30)

	init=Init(mesh,η,v)

	plt = plot(layout=(1,2))
	plot!(plt[1,1], mesh.x, [η v];
     title=string("initial data"),
     label=["η" "u"])

    plot!(plt[1,2], fftshift(mesh.k),
     [log10.(abs.(fftshift(fft(η)))) log10.(abs.(fftshift(fft(v))))];
     title="frequency",
	 label=["η" "u"])
	 display(plt)

	model  = WhithamBoussinesq(param)
	problem = Problem(model, init, param);

	solve!( problem )
	ηf = model.mapfro(problem.data.U[end])[1]
	p = plot(layout=(1,2))
	plot!(p[1,1],mesh.x,[η ηf];
	title="surface deformation",
	label=["initial" "final"])
	plot!(p[1,2],mesh.x,η-ηf;
	title="difference",
	label="")
	display(p)

	anim = create_animation( problem )
	gif(anim, fps=15)
end
nothing
