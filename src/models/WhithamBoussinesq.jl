export WhithamBoussinesq
using FFTW
"""
    WhithamBoussinesq(params;kwargs)

Define an object of type `AbstractModel` in view of solving the initial-value problem for
a Boussinesq-type model with full-dispersion property.

# Argument
`param` is of type `NamedTuple` and must contain

- dimensionless parameters `ϵ` (nonlinearity) and `μ` (dispersion);
- numerical parameters to construct the mesh of collocation points as `mesh = Mesh(param)`
- a parameter `α` which determines the model solved:
    - If `α = 1`, then the model has been introduced and studied by E. Dinvay and collaborators;
    - If `α = 1/2`, then the model is a quasilinear version;
    - If `α < 1/2`, then expect instabilities stemming from ill-posedness of the model.

## Keywords
- `ktol`: tolerance of the low-pass Krasny filter (default is `0`, i.e. no filtering);
- `dealias`: dealiasing with Orlicz rule `1-dealias/(dealias+2)` (default is `0`, i.e. no dealiasing).
- `verbose`: prints information if `true` (default is `true`).


# Return values
Generate necessary ingredients for solving an initial-value problem via `solve!` and in particular
1. a function `WhithamBoussinesq.f!` to be called in the time-integration solver;
2. a function `WhithamBoussinesq.mapto` which from `(η,v)` of type `InitialData` provides the raw data matrix on which computations are to be executed.
3. a function `WhithamBoussinesq.mapfro` which from such data matrix returns the Tuple of real vectors `(η,v)`, where
    - `η` is the surface deformation;
    - `v` is the derivative of the trace of the velocity potential.

"""
mutable struct WhithamBoussinesq <: AbstractModel

	label   :: String
	f!		:: Function
	mapto	:: Function
	mapfro	:: Function
	param	:: NamedTuple
	kwargs	:: NamedTuple

    function WhithamBoussinesq(param::NamedTuple;
								dealias=0,ktol=0,verbose=true)

		label = string("Whitham-Boussinesq")
		μ 	= param.μ
		ϵ 	= param.ϵ
		α	= param.α
		mesh = Mesh(param)
		param = ( ϵ = ϵ, μ = μ, α = α, xmin = mesh.xmin, xmax = mesh.xmax, N = mesh.N )
		kwargs = (dealias=dealias,ktol=ktol,verbose=verbose)

		x 	= mesh.x
        F₁ 	= tanh.(sqrt(μ)*abs.(mesh.k))./(sqrt(μ)*abs.(mesh.k))
		F₁[1] 	= 1
		F₂ = F₁.^α
    	∂ₓ	=  1im * mesh.k
		K = mesh.kmax * (1-dealias/(2+dealias))
		Π⅔ 	= abs.(mesh.k) .<= K # Dealiasing low-pass filter
		if dealias == 0
			if verbose @info "no dealiasing" end
			Π⅔ 	= ones(size(mesh.k))
		elseif verbose
			@info string("dealiasing : spectral scheme for power ", dealias + 1," nonlinearity ")
		end
        η = zeros(Float64, mesh.N)
        v = zeros(Float64, mesh.N)
		fftη = zeros(Complex{Float64}, mesh.N)
        fftv = zeros(Complex{Float64}, mesh.N)

		# Evolution equations are ∂t U = f!(U)
		function f!(U)

		    fftv .= U[:,2]
			fftη .= F₂.*U[:,2]
			v .= real(ifft(fftη))
			fftη .= U[:,1]
		   	η .= real(ifft(U[:,1]))

		   	U[:,1] .= -∂ₓ.*(F₁.*fftv.+ϵ*Π⅔.*F₂.*fft(η.*v))
		   	U[:,2] .= -∂ₓ.*(fftη.+ϵ/2*Π⅔.*fft(v.^2))
			U[abs.(U).< ktol ].=0

		end

		# Discrete Fourier transform with, possibly, dealiasing and Krasny filter.
		function mapto(data::InitialData)
			U = [Π⅔ .* fft(data.η(x)) Π⅔ .*fft(data.v(x))]
			U[abs.(U).< ktol ].=0
			return U
		end

		# Return `(η,v)`, where
		# - `η` is the surface deformation;
		# - `v` is the derivative of the trace of the velocity potential.
		# Inverse Fourier transform and takes the real part.
		function mapfro(U)
			real(ifft(U[:,1])),real(ifft(U[:,2]))
		end

        new(label, f!, mapto, mapfro, param, kwargs)
    end
end
