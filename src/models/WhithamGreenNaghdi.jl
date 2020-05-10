export WhithamGreenNaghdi,mapto,mapfro
using LinearMaps,IterativeSolvers

"""
    WhithamGreenNaghdi(param;kwargs)

Defines an object of type `AbstractModel` in view of solving the initial-value problem for
the modified Green-Naghdi model proposed by V. Duchêne, S. Israwi and R. Talhouk.

# Argument
`param` is of type `NamedTuple` and must contain
- dimensionless parameters `ϵ` (nonlinearity) and `μ` (dispersion);
- numerical parameters to construct the mesh of collocation points as `mesh = Mesh(param)`
E.g. `param = ( μ  = 0.1, ϵ  = 1, N  = 2^9, L  = 10*π)`

## Keywords
- `SGN :: Bool`: if `true` computes the Serre-Green-Naghdi (SGN) instead of Whitham-Green-Naghdi (WGN) system (default is `false`);
- `iterative :: Bool`: solves the elliptic problem through GMRES if `true`, LU decomposition if `false` (default is `true`);
- `precond :: Bool`: Preconditioner of GMRES is based on WGN if `true`, SGN otherwise (default is `true`);
- `gtol :: Real`: relative tolerance of the GMRES algorithm (default is `1e-14`);
- `ktol :: Real`: tolerance of the Krasny filter (default is `0`, i.e. no filtering);
- `dealias :: Int`: dealiasing with Orlicz rule `1-dealias/(dealias+2)` (default is `0`, i.e. no dealiasing).

# Return values
This generates
1. a function `WhithamGreenNaghdi` to be called in the time-integration solver;
2. a function `mapto` which from `(η,v)` of type `InitialData` provides the  data matrix on which computations are to be executed.
3. a function `mapfro` which from such data matrix returns the Tuple of real vectors `(η,v,u)`, where

    - `η` is the surface deformation;
    - `v` is the derivative of the trace of the velocity potential;
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


    # label   :: String
	# datasize:: Int
	# param	:: NamedTuple
	# kwargs  :: NamedTuple
	# μ 		:: Real
	# ϵ 		:: Real
	# x   	:: Vector{Float64}
    # F₀   	:: Vector{Complex{Float64}}
    # ∂ₓ      :: Vector{Complex{Float64}}
    # Π⅔      :: Vector{Float64}
	# Id 	    :: BitArray{2}
	# FFT 	:: Array{Complex{Float64},2}
	# IFFT 	:: Array{Complex{Float64},2}
	# IFFTF₀ 	:: Array{Complex{Float64},2}
	# M₀      :: Array{Complex{Float64},2}
    # h    	:: Vector{Complex{Float64}}
	# u    	:: Vector{Complex{Float64}}
	# fftv    :: Vector{Complex{Float64}}
	# fftη   	:: Vector{Complex{Float64}}
	# fftu  	:: Vector{Complex{Float64}}
	# hdu    	:: Vector{Complex{Float64}}
	# L   	:: Array{Complex{Float64},2}
	# Precond :: Any
	# iterate :: Bool
	# ktol 	:: Real
	# gtol 	:: Real


    function WhithamGreenNaghdi(param::NamedTuple;iterate=true,SGN=false,dealias=0,ktol=0,gtol=1e-14,precond=true)
		if SGN == true
			label = string("Serre-Green-Naghdi")
		else
			label = string("Whitham-Green-Naghdi")
		end
		@info label

		kwargs = (iterate=iterate,SGN=SGN,dealias=dealias,ktol=ktol,gtol=gtol,precond=precond)
		μ 	= param.μ
		ϵ 	= param.ϵ
		mesh = Mesh(param)
		k = mesh.k
		x 	= mesh.x
		x₀ = mesh.x[1]

		∂ₓ	=  1im * mesh.k
		F₁ 	= tanh.(sqrt(μ)*abs.(k))./(sqrt(μ)*abs.(k))
		F₁[1] 	= 1                 # Differentiation
		if SGN == true
	                F₀ = sqrt(μ)*∂ₓ
	    else
	                F₀ = 1im * sqrt.(3*(1 ./F₁ .- 1)).*sign.(k)
		end
		if precond == true
			Precond = Diagonal( 1 ./  F₁ )
		else
			Precond = Diagonal( 1 .+ μ/3*k.^2 )
			#Precond = lu( exp.(-1im*k*x') ) # #Diagonal( ones(size(k)) )
		end
		K = mesh.kmax * (1-dealias/(2+dealias))
		Π⅔ 	= abs.(mesh.k) .<= K # Dealiasing low-pass filter
		if dealias == 0
			@info "no dealiasing"
			Π⅔ 	= ones(size(mesh.k))
		else
			@info "dealiasing"
		end
		if iterate == true
			@info "GMRES method"
		else
			@info "LU decomposition"
		end
		FFT = exp.(-1im*k*(x.-x₀)');
        IFFT = exp.(1im*k*(x.-x₀)')/length(x);
		M₀ = IFFT * Diagonal( F₀ ) * FFT
		IFFTF₀ = IFFT * Diagonal( F₀ )
        Id = Diagonal(ones(size(x)));
		h = zeros(Complex{Float64}, mesh.N)
		u, fftv, fftη, fftu, hdu = (similar(h),).*ones(5)
		L = similar(FFT)

		function f!(U)
			fftη .= U[:,1]
			h .= 1 .+ ϵ*ifft(fftη)
			fftv .= U[:,2]
			#fftv[abs.(fftv).< ktol ].=0   # Krasny filter
			if iterate == false
				L .= Id - 1/3 * FFT * Diagonal( 1 ./h ) * M₀ * Diagonal( h.^3 ) * IFFTF₀
				fftu .= L \ fftv
			elseif iterate == true
		        function LL(hatu)
		            hatu- 1/3 *Π⅔.*fft( 1 ./h .* ifft( F₀ .* Π⅔.*fft( h.^3 .* ifft( F₀ .* hatu ) ) ) )
				end
				fftu .= gmres( LinearMap(LL, length(h); issymmetric=false, ismutating=false) , fftv ;
						Pl = Precond,
						tol = gtol )
			end
			u .= ifft(fftu)
			hdu .= h .* ifft(Π⅔.*F₀.*fftu)
		   	U[:,1] .= -∂ₓ.*Π⅔.*(fftu .+ ϵ * fft(ifft(fftη) .* u))
			U[:,2] .= -∂ₓ.*Π⅔.*(fftη .+ ϵ * fft( u.*ifft(fftv)
							.- 1/2 * u.^2 .- 1/2 * hdu.^2 ) )
			U[abs.(U).< ktol ].=0
		end

		"""
		    mapto(WhithamGreenNaghdi, data)
		`data` is of type `InitialData`, maybe constructed by `Init(...)`.

		Performs a discrete Fourier transform with, possibly, dealiasing and Krasny filter.

		See documentation of `WhithamGreenNaghdi` for more details.

		"""
		function mapto(data::InitialData)
			U = [Π⅔ .* fft(data.η(x)) Π⅔ .*fft(data.v(x))]
			U[abs.(U).< ktol ].=0
			return U
		end

		"""
		    mapfro(WhithamGreenNaghdi, data)
		`data` is of type `Array{Complex{Float64},2}`, e.g. `last(p.data.U)` where `p` is of type `Problem`.

		Returns `(η,v)`, where
		- `η` is the surface deformation;
		- `v` is the derivative of the trace of the velocity potential.

		Performs an inverse Fourier transform and takes the real part.

		See documentation of `WhithamGreenNaghdi` for more details.
		"""
		function mapfro(U)
			real(ifft(U[:,1])),real(ifft(U[:,2]))
		end
		"""
		    mapfrofull(WhithamGreenNaghdi, data)
		`data` is of type `Array{Complex{Float64},2}`, e.g. `last(p.data.U)` where `p` is of type `Problem`.

		Returns `(η,v,u)`, where
		- `η` is the surface deformation;
		- `v` is the derivative of the trace of the velocity potential;
		- `u` corresponds to the layer-averaged velocity.

		Performs an inverse Fourier transform and take the real part, plus solves the costly elliptic problem for `u`.

		See documentation of `WhithamGreenNaghdi` for more details.
		"""
		function mapfrofull(U)
				fftη .= U[:,1]
			   	h .= 1 .+ ϵ*ifft(fftη)
				L .= Id - 1/3 * FFT * Diagonal( 1 ./h ) * M₀ * Diagonal( h.^3 ) * IFFTF₀

				   real(ifft(U[:,1])),real(ifft(U[:,2])),real(ifft(L \ U[:,2]))
		end

        new(label, f!, mapto, mapfro, mapfrofull, param, kwargs)
    end
end
