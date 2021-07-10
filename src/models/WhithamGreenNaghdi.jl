export WhithamGreenNaghdi
using FFTW,LinearAlgebra,LinearMaps,IterativeSolvers

"""
    WhithamGreenNaghdi(param;kwargs)

Define an object of type `AbstractModel` in view of solving the initial-value problem for
the modified Green-Naghdi model proposed by V. Duchêne, S. Israwi and R. Talhouk.

# Argument
`param` is of type `NamedTuple` and must contain
- dimensionless parameters `ϵ` (nonlinearity) and `μ` (dispersion);
- numerical parameters to construct the mesh of collocation points as `mesh = Mesh(param)`.

## Keywords
- `SGN`: if `true` computes the Serre-Green-Naghdi (SGN) instead of Whitham-Green-Naghdi (WGN) system (default is `false`);
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
1. a function `WhithamGreenNaghdi.f!` to be called in the time-integration solver;
2. a function `WhithamGreenNaghdi.mapto` which from `(η,v)` of type `InitialData` provides the raw data matrix on which computations are to be executed;
3. a function `WhithamGreenNaghdi.mapfro` which from such data matrix returns the Tuple of real vectors `(η,v)`, where

    - `η` is the surface deformation;
    - `v` is the derivative of the trace of the velocity potential;
4. additionally, a handy function `WhithamGreenNaghdi.mapfrofull` which from data matrix returns the Tuple of real vectors `(η,v,u)`, where

	- `u` corresponds to the layer-averaged velocity.

"""
mutable struct WhithamGreenNaghdi <: AbstractModel

	label   :: String
	f!		:: Function
	mapto	:: Function
	mapfro	:: Function
	mapfrofull	:: Function
	param	:: NamedTuple
	kwargs  :: NamedTuple

    function WhithamGreenNaghdi(param::NamedTuple;SGN=false,dealias=0,ktol=0,iterate=true,gtol=1e-14,precond=true,restart=nothing,maxiter=nothing,verbose=true)
		if SGN == true
			label = string("Serre-Green-Naghdi")
		else
			label = string("Whitham-Green-Naghdi")
		end
		if verbose @info string("model ",label) end

		kwargs = (iterate=iterate,SGN=SGN,dealias=dealias,ktol=ktol,gtol=gtol,precond=precond,verbose=verbose)
		μ 	= param.μ
		ϵ 	= param.ϵ
		mesh = Mesh(param)
		param = ( ϵ = ϵ, μ = μ, xmin = mesh.xmin, xmax = mesh.xmax, N = mesh.N )

		k = mesh.k
		x 	= mesh.x
		x₀ = mesh.x[1]

		∂ₓ	=  1im * mesh.k
		F₁ 	= tanh.(sqrt(μ)*abs.(k))./(sqrt(μ)*abs.(k))
		F₁[1] 	= 1
		if SGN == true
			F₀ = sqrt(μ)*∂ₓ
	    else
			F₀ = 1im * sqrt.(3*(1 ./F₁ .- 1)).*sign.(k)
		end
		if precond == true
			Precond = Diagonal( 1 .+ μ/3*k.^2 ) #Diagonal( 1 ./  F₁ )
		elseif precond == false
			Precond = Diagonal( ones(size(k)) )
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
		M₀ = IFFT * Diagonal( F₀ .* Π⅔)* FFT
		IFFTF₀ = IFFT * Diagonal( F₀ .* Π⅔)
        Id = Diagonal(ones(size(x)));
		h = zeros(Complex{Float64}, mesh.N)
		u, fftv, fftη, fftu, hdu = (similar(h),).*ones(5)
		L = similar(FFT)
		if maxiter == nothing maxiter = mesh.N end
		if restart == nothing restart = min(20,mesh.N) end


		# Evolution equations are ∂t U = f!(U)
		function f!(U)
			fftη .= U[:,1]
			h .= 1 .+ ϵ*ifft(fftη)
			fftv .= U[:,2]
			if iterate == false
				L .= Id - 1/3 * Diagonal(Π⅔) * FFT * Diagonal( 1 ./h ) * M₀ * Diagonal( h.^3 ) * IFFTF₀
				fftu .= L \ fftv
			elseif iterate == true
		        function LL(hatu)
		            hatu- 1/3 *Π⅔.*fft( 1 ./h .* ifft( F₀ .* Π⅔.*fft( h.^3 .* ifft( F₀ .* Π⅔.* hatu ) ) ) )
				end
				fftu .= gmres( LinearMap(LL, length(h); issymmetric=false, ismutating=false) , fftv ;
						restart = restart, maxiter = maxiter, Pl = Precond, reltol = gtol )
			end
			u .= ifft(fftu)
			hdu .= h .* ifft(Π⅔.*F₀.*fftu)
		   	U[:,1] .= -∂ₓ.*Π⅔.*(fftu .+ ϵ * fft(ifft(fftη) .* u))
			U[:,2] .= -∂ₓ.*Π⅔.*(fftη .+ ϵ * fft( u.*ifft(fftv)
							.- 1/2 * u.^2 .- 1/2 * hdu.^2 ) )
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
		function mapfrofull(U)
				fftη .= U[:,1]
			   	h .= 1 .+ ϵ*ifft(fftη)
				L .= Id - 1/3 * Diagonal(Π⅔) * FFT * Diagonal( 1 ./h ) * M₀ * Diagonal( h.^3 ) * IFFTF₀

				   real(ifft(U[:,1])),real(ifft(U[:,2])),real(ifft(L \ U[:,2]))
		end

        new(label, f!, mapto, mapfro, mapfrofull, param, kwargs)
    end
end
