export IsobeKakinuma
using FFTW,LinearAlgebra,LinearMaps,IterativeSolvers

"""
    IsobeKakinuma(param;kwargs)

Define an object of type `AbstractModel` in view of solving the initial-value problem for
the Isobe-Kakinuma model.

# Argument
`param` is of type `NamedTuple` (or a collection `NamedTuple`s) of and must contain
- dimensionless parameters `ϵ` (nonlinearity) and `μ` (dispersion);
- numerical parameters to construct the mesh of collocation points as `mesh = Mesh(param)`.

## Keywords
- `iterative`: solve the elliptic problem through GMRES if `true`, LU decomposition if `false` (default is `true`);
- `precond`: use a (left) preconditioner for GMRES if `true` (default), choose `precond` as the preconditioner if provided;
- `gtol`: relative tolerance of the GMRES algorithm (default is `1e-14`);
- `restart`: the corresponding option of the GMRES algorithm (default is `100`);
- `maxiter`: the corresponding option of GMRES (default is `nothing`);
- `ktol`: tolerance of the Krasny filter (default is `0`, i.e. no filtering);
- `dealias`: dealiasing with Orlicz rule `1-dealias/(dealias+2)` (default is `0`, i.e. no dealiasing);
- `verbose`: prints information if `true` (default is `true`).

# Return values
Generate necessary ingredients for solving an initial-value problem via `solve!` and in particular
1. a function `IsobeKakinuma.f!` to be called in the time-integration solver;
2. a function `IsobeKakinuma.mapto` which from `(η,v)` of type `InitialData` provides the raw data matrix on which computations are to be executed;
3. a function `IsobeKakinuma.mapfro` which from such data matrix returns the Tuple of real vectors `(η,v)`, where

    - `η` is the surface deformation;
    - `v` is the derivative of the trace of the velocity potential;
4. additionally, a handy function `IsobeKakinuma.mapfrofull` which from data matrix returns the Tuple of real vectors `(η,v,Φ)`, where

	- `Φ` is the Vector of the basis functions `ϕi` (`i∈{0,...,N}`).

"""
mutable struct IsobeKakinuma <: AbstractModel

	label   :: String
	f!		:: Function
	mapto	:: Function
	mapfro	:: Function
	mapfrofull	:: Function
	param	:: NamedTuple
	kwargs  :: NamedTuple

    function IsobeKakinuma(param::NamedTuple;dealias=0,ktol=0,iterate=true,gtol=1e-14,precond=false,restart=nothing,maxiter=nothing,verbose=true)
		label = string("Isobe-Kakinuma")
		if verbose @info string("model ",label) end

		kwargs = (iterate=iterate,dealias=dealias,ktol=ktol,gtol=gtol,precond=precond,verbose=verbose)
		μ 	= param.μ
		ϵ 	= param.ϵ
		mesh = Mesh(param)
		param = ( ϵ = ϵ, μ = μ, xmin = mesh.xmin, xmax = mesh.xmax, N = mesh.N )

		k = mesh.k
		x 	= mesh.x
		x₀ = mesh.x[1]

		∂ₓ	=  1im * mesh.k
		if precond == true
			Precond = Diagonal( [-1 .- 2*μ/5*k.^2 ; 1 .+ 2*μ/5*k.^2] ) #Diagonal( 1 ./  F₁ )
			#Precond = [ Diagonal( -μ/2*k.^2)  Diagonal(1 .- μ/10 * k.^2 ) ;
			#		 Diagonal( ones( size(k) ) ) Diagonal( ones( size(k) ) ) ]
		elseif precond == false
			Precond = Diagonal( ones( size( [k;k])) )
		else
			Precond = precond
		end
		K = mesh.kmax * (1-dealias/(2+dealias))
		Π⅔ 	= abs.(mesh.k) .<= K # Dealiasing low-pass filter
		if dealias == 0
			if verbose @info "no dealiasing" end
			Π⅔ 	= ones(size(mesh.k))
		elseif verbose
			@info string("dealiasing : spectral scheme for power ", dealias + 1," nonlinearity ")
		end
		if iterate == true && verbose
			@info "elliptic problem solved with GMRES method"
		elseif verbose
			@info "elliptic problem solved with LU decomposition"
		end
		FFT = exp.(-1im*k*(x.-x₀)');
        IFFT = exp.(1im*k*(x.-x₀)')/length(x);
        Id = Diagonal(ones(size(x)));
		h = zeros(Complex{Float64}, mesh.N)
		u, fftv, fftη, u, w, G = (similar(h),).*ones(6)
		C = similar([h ; h])
		CC = reshape(C,:,2)
		fftϕ = similar(CC)
		L = similar([FFT FFT ; FFT FFT])
		if maxiter == nothing maxiter = mesh.N end
		if restart == nothing restart = min(20,mesh.N) end


		# Evolution equations are ∂t U = f!(U)
		function f!(U)
			fftη .= U[:,1]
			h .= 1 .+ ϵ*ifft(fftη)
			fftv .= U[:,2]
			C .= [-μ/2 * ∂ₓ.* fftv ; zero(fftv)]
			if iterate == false
				L .= [μ/2*Diagonal( ∂ₓ.^2 .* Π⅔) (Id + μ/10 * Diagonal(Π⅔) * FFT * Diagonal( h.^2 ) * IFFT * Diagonal( ∂ₓ.^2 .* Π⅔)) ;
						Id (FFT * Diagonal( h.^2 ) * IFFT .* Π⅔) ]

				fftϕ .= reshape(L \ C,:,2)
			elseif iterate == true # does not work yet
				CC .= reshape(C,:,2)
		        function LL(fftϕ)
		            [μ/2*∂ₓ.^2 .* Π⅔.* fftϕ[1:end÷2] .+ fftϕ[end÷2+1:end] .+ μ/10*Π⅔.*fft( (h.^2) .* ifft( ∂ₓ.^2 .* Π⅔ .* fftϕ[end÷2+1:end] ) ) ;
					fftϕ[1:end÷2] + fft( (h.^2) .* ifft(  Π⅔ .* fftϕ[end÷2+1:end])) ]
				end
				fftϕ .= reshape( gmres( LinearMap(LL, 2*length(x); issymmetric=false, ismutating=false) , C ;
						restart = restart, maxiter = maxiter, Pl = Precond, reltol = gtol ) , : ,2)
			end
			u .= ifft( ∂ₓ.* fftϕ[:,1] .+ fftv) .+ (h.^2)  .* ifft(∂ₓ.* fftϕ[:,2])
			w .= h.* ifft( fftϕ[:,2])
			G .= -∂ₓ.*Π⅔.* fft( h.* ifft( ∂ₓ.* fftϕ[:,1] .+ fftv) .+ (h.^3)/3  .* ifft(∂ₓ.* fftϕ[:,2]) )

		   	U[:,1] .= G
			U[:,2] .= -∂ₓ.*Π⅔.*(fftη .+ ϵ * fft( h.* w .* ifft(-G)
							.+ 1/2 * (u.^2 .+ 1/μ * w.^2 ) ) )
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
		# Returns `(η,v,u)`, where
		# - `η` is the surface deformation;
		# - `v` is the derivative of the trace of the velocity potential;
		# - `u` corresponds to the layer-averaged velocity.
		# Inverse Fourier transform and take the real part, plus solves the costly elliptic problem for `u`.
		function mapfrofull(U)   # remains to be done
				fftη .= U[:,1]
			   	h .= 1 .+ ϵ*ifft(fftη)
				L .= Id - 1/3 * Diagonal(Π⅔) * FFT * Diagonal( 1 ./h ) * M₀ * Diagonal( h.^3 ) * IFFTF₀

				   real(ifft(U[:,1])),real(ifft(U[:,2])),real(ifft(L \ U[:,2]))
		end

        new(label, f!, mapto, mapfro, mapfrofull, param, kwargs)
    end
end
