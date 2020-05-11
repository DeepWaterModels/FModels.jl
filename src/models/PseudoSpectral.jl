export PseudoSpectral
using FFTW
"""
    PseudoSpectral(param;kwargs)

Define an object of type `AbstractModel` in view of solving the initial-value problem for
the modified water waves expansion proposed by West et al., Craig-Sulem, etc.

# Argument
`param` is of type `NamedTuple` and must contain
- dimensionless parameters `ϵ` (nonlinearity) and `μ` (dispersion);
- numerical parameters to construct the mesh of collocation points as `mesh = Mesh(param)`

## Keywords
- `order :: Int`: the order of the expansion; linear system if `1`, quadratic if `2`, cubic if `3`, quartic if `4` (default and other values yield `2`);
- `lowpass`: parameter of smooth regularization operator (default is `0`, i.e. no regularization);
- `ktol`: tolerance of the low-pass Krasny filter (default is `0`, i.e. no filtering);
- `dealias`: dealiasing with Orlicz rule `1-dealias/(dealias+2)` (default is `0`, i.e. no dealiasing);
- `verbose`: prints information if `true` (default is `true`).

# Return values
Generate necessary ingredients for solving an initial-value problem via `solve!` and in particular
1. a function `PseudoSpectral.f!` to be called in the time-integration solver;
2. a function `PseudoSpectral.mapto` which from `(η,v)` of type `InitialData` provides the raw data matrix on which computations are to be executed;
3. a function `PseudoSpectral.mapfro` which from such data matrix returns the Tuple of real vectors `(η,v)`, where
    - `η` is the surface deformation;
    - `v` is the derivative of the trace of the velocity potential.

"""
mutable struct PseudoSpectral <: AbstractModel

    label   :: String
	f!		:: Function
	mapto	:: Function
	mapfro	:: Function
	param	:: NamedTuple
	kwargs	:: NamedTuple

    function PseudoSpectral(param::NamedTuple;
							order=2,lowpass=0,ktol=0,dealias=0, verbose=true)

		if order in [1,2,3,4] && verbose
			@info string("Pseudo-spectral model with nonlinearity of order ", order)
		elseif verbose
			order = 2
			@info string("Pseudo-spectral model with nonlinearity of order ", order)
		end
		label = string("WW",order)

		μ 	= param.μ
		ϵ 	= param.ϵ
		n 	= order
		mesh = Mesh(param)

		param = ( ϵ = ϵ, μ = μ, xmin = mesh.xmin, xmax = mesh.xmax, N = mesh.N )
		kwargs = (order=order,lowpass=lowpass,dealias=dealias,ktol=ktol,verbose=verbose)

		x = mesh.x
		k = copy(mesh.k)
		cutoff = k -> (1 + tanh(-abs(k)+1/abs(k)))/2
		Π = cutoff.(lowpass*k)   # regularisation cutoff
		K = mesh.kmax * (1-dealias/(2+dealias))
		Π⅔ 	= abs.(mesh.k) .<= K # Dealiasing low-pass filter
		Π⅔ 	= abs.(mesh.k) .<= K # Dealiasing low-pass filter
		if dealias == 0
			if verbose @info "no dealiasing" end
			Π⅔ 	= ones(size(mesh.k))
		elseif verbose
			@info string("dealiasing : spectral scheme for power ", dealias + 1," nonlinearity ")
		end
        F₀ 	= tanh.(sqrt(μ)*abs.(mesh.k))./(sqrt(μ)*abs.(mesh.k))
		F₀[1] 	= 1
		G₀ 	= sqrt(μ)*abs.(mesh.k).*tanh.(sqrt(μ)*abs.(mesh.k))
		∂ₓ	=  1im * sqrt(μ)* mesh.k            # Differentiation
		z = zeros(Complex{Float64}, mesh.N)
		η = copy(z) ; v = copy(z) ; Lphi = copy(z) ; LzLphi = copy(z) ; dxv = copy(z) ;
		fftη = copy(z) ; fftv = copy(z) ; Q = copy(z) ; R = copy(z) ;

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

		# Evolution equations are ∂t U = f!(U)
		function f!(U)

			fftη .= U[:,1]
		    fftv .= U[:,2]
			Q .= -∂ₓ.*F₀.*fftv
			R .= -fftη

			# attention,  G₀=-L dans Choi
			if n >= 2
				η  .= ifft(Π.*U[:,1])
				v  .= ifft(U[:,2])
				Lphi .= -ifft(Q)
				Q += -ϵ *∂ₓ.*fft(η.*ifft(fftv)) .+ ϵ*G₀.*fft(η.*Lphi)
				R += ϵ/2*fft(-v.^2 .+ Lphi.^2)
			end
			if n >= 3
				LzLphi .= ifft(-G₀.* fft(η.*Lphi))
				dxv .= ifft(∂ₓ.*fftv)
				Q += ϵ^2*G₀.*fft(η.* LzLphi + 1/2 * η.^2 .* dxv ) .- ϵ^2*∂ₓ.*∂ₓ.*fft( 1/2 * η.^2  .*Lphi)
				R += ϵ^2*fft(Lphi .* ( LzLphi .+ η.* dxv ) )
			end
			if n >= 4
				Q += ϵ^3 * G₀.*fft(η.*ifft(-G₀.* fft(η.*LzLphi + 1/2 * η.^2 .* dxv ) )
							.+ 1/2 * η.^2 .* ifft(∂ₓ.*∂ₓ.*fft( η  .* Lphi ) )
							.- 1/6 * η.^3 .* ifft(∂ₓ.*∂ₓ.*fft( Lphi ) ) ) .-
					   ϵ^3 * ∂ₓ.*∂ₓ.*fft( 1/2 * η.^2  .*LzLphi .+ 1/3 * η.^3  .* dxv )
				R += ϵ^3 * fft( Lphi .*  ifft(-G₀.* fft(η.*LzLphi + 1/2 * η.^2 .* dxv ) )
						.+ 1/2* (LzLphi .+ η .* dxv ).^2
						.+ 1/2* η.* Lphi.^2 .* ifft(∂ₓ.*∂ₓ.* fftη)
						.- 1/2* (η.^2).* (ifft(∂ₓ .* fft(Lphi))).^2 ) .+
						1/4*ϵ^3 * ∂ₓ.*∂ₓ.*fft((η .* Lphi).^2)
			end
		   	U[:,1] .= Π⅔.*Q/sqrt(μ)
		   	U[:,2] .= Π⅔.*∂ₓ.*Π.*R/sqrt(μ)
			U[abs.(U).< ktol ].=0

		end


        new(label, f!, mapto, mapfro, param, kwargs )
    end
end
