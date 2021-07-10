export WaterWaves
#using Statistics #only for function mean in WaterWaves.jl...
using FFTW,LinearAlgebra

function cotanh( x )
	y=1 ./ tanh.(x+(x.==0))
	y[x.==0].=0
	return y
end
function mean(x)
	sum(x)/length(x)
end

"""
    WaterWaves(params)

Define an object of type `AbstractModel` in view of solving the water waves system
(via conformal mapping).

# Argument
`param` is of type `NamedTuple` and must contain
- dimensionless parameters `ϵ` (nonlinearity) and `μ` (dispersion);
- numerical parameters to construct the mesh of collocation points as `mesh = Mesh(param)`.

# Return values
Generate necessary ingredients for solving an initial-value problem via `solve!` and in particular
1. a function `WaterWaves.f!` to be called in the time-integration solver;
2. a function `WaterWaves.mapto` which from `(η,v)` of type `InitialData` provides the raw data matrix on which computations are to be executed;
3. a function `WaterWaves.mapfro` which from such data matrix returns the Tuple of real vectors `(x,η,v)`, where
	- `x` is a vector of collocation points (non-regularly spaced);
    - `η` is the surface deformation at points `x`;
    - `v` is the derivative of the trace of the velocity potential at points `x`.

"""
mutable struct WaterWaves <: AbstractModel

    label   :: String
	f!		:: Function
	f1!		:: Function
	f2!		:: Function
	mapto	:: Function
	mapfro	:: Function
	param	:: NamedTuple
	kwargs  :: NamedTuple


    function WaterWaves(param::NamedTuple;dealias=1,tol=1e-16,maxiter=1000,verbose=true)

		label = "water waves"
		kwargs = (dealias=dealias,tol=tol,maxiter=maxiter,verbose=verbose)

		μ 	= param.μ
		ϵ 	= param.ϵ
		mesh = Mesh(param)

		x 	= mesh.x
		k 	= mesh.k

    	∂ₓ	=  1im * mesh.k            # Differentiation
		K = mesh.kmax * (1-dealias/(2+dealias))
		Π⅔ 	= abs.(mesh.k) .<= K # Dealiasing low-pass filter
		if dealias == 0
			if verbose @info "no dealiasing" end
			Π⅔ 	= ones(size(mesh.k))
		elseif verbose
			@info string("dealiasing : spectral scheme for power ", dealias + 1," nonlinearity ")
		end

        z = zeros(Float64, mesh.N)
        phi = zeros(Float64, mesh.N)
		ξ = zeros(Float64, mesh.N)
		xv = zeros(Float64, mesh.N)
		Dxv = zeros(Float64, mesh.N)
		Dz = zeros(Float64, mesh.N)
		Dphi = zeros(Float64, mesh.N)
		J = zeros(Float64, mesh.N)
		M1 = zeros(Float64, mesh.N)
		M2 = zeros(Float64, mesh.N)
		q0 = 0

		# Constructs conformal variables in the flat strip
		function mapto(data::InitialData)

			function iter( u )
			    return real.(ifft(-sqrt(μ)*1im*Π⅔ .*cotanh(sqrt(μ)*(1+ϵ*mean(data.η(x+ϵ*u)))*k).*fft(data.η(x+ϵ*u))))
			end

			# on démarre avec deux coefficients spécifiques:
			δ = mean(data.η(x))
			u0 = real.(ifft(-sqrt(μ)*1im*Π⅔ .*cotanh(sqrt(μ)*k).*fft(data.η(x))))

			# détermination du point fixe de la fonction contractante:
			norm0=norm(data.η(x))
			niter=0
			nu0=u0 .+ 1
			while  (norm(nu0-u0)>tol*norm0 && niter<maxiter)
			    nu0=u0
			    u0 = iter(nu0)
			    niter+=1
			end
			if verbose
				if niter==maxiter
				    @warn "The fix point algorithm did not converge"
				    @warn string("Estimated normalized error : ",norm(nu0-u0)/norm0)
				else
					@info string("The fix point algorithm converged in ",niter," iterations")
				end
		end
			δ = mean(data.η(x+ϵ*u0))

			# Obtention des conditions initiales dans le domaine redressé:
			z0 = data.η(x+ϵ*u0)
			v0 = data.v(x+ϵ*u0) .* (1 .+ ϵ*real.(ifft(Π⅔.*∂ₓ.*fft(u0))) )

			[z0 v0] ######
		end

		# Reconstructs physical variables from conformal variables
		function mapfro(U)
			z  .= U[:,1]
		   	ξ  .= real.(sqrt(μ)*(1+ϵ*mean(z))*k)
	       	xv .= real.(-1im*sqrt(μ)*ifft( cotanh(ξ) .* fft( z )))

		   return x + ϵ*xv, real.(z) , real.( U[:,2] )
		end

		# Water Waves equations are ∂t U = f!(U)
		function f!(U)
		   	z .= U[:,1]
		   	Dphi .= U[:,2]
			ξ .= sqrt(μ)*(1 .+ ϵ*mean(z))*k
			#xv .=real.(ifft( Π⅔ .* cotanh(ξ) .* fft( z )))
			Dxv .= real.(sqrt(μ)*ifft(-1im* Π⅔ .* ∂ₓ.* cotanh(ξ) .* fft(z)))
			Dz .= real.(ifft(Π⅔ .* ∂ₓ.*fft(z)))

			J .= (1 .+ ϵ*Dxv).^2 + μ*(ϵ*Dz).^2 # Jacobien
			M1 = real.(-1im/sqrt(μ)*ifft(Π⅔ .* tanh.(ξ).* fft(Dphi)))
			M2 = real.( 1im*sqrt(μ)*ifft(Π⅔ .* cotanh(ξ) .* fft(M1./J )))
			q0 = mean((1 .+ ϵ*Dxv).*M2 + ϵ*μ*Dz.*M1./J)

			U[:,1] .= (1 .+ ϵ*Dxv).*M1./J - ϵ*Dz.*M2 + ϵ*q0*Dz
			U[:,2] .= real.(-ifft(∂ₓ .* fft( z +ϵ*Dphi.*M2 + ϵ/2 .*(Dphi.^2 -μ*M1.^2)./J - ϵ*q0*Dphi)))

		end
		function f1!(U1,U2)
		   	z .= U1
		   	Dphi .= U2
			ξ .= sqrt(μ)*(1 .+ ϵ*mean(z))*k
			#xv .=real.(ifft( Π⅔ .* cotanh(ξ) .* fft( z )))
			Dxv .= real.(sqrt(μ)*ifft(-1im* Π⅔ .* ∂ₓ.* cotanh(ξ) .* fft(z)))
			Dz .= real.(ifft(Π⅔ .* ∂ₓ.*fft(z)))

			J .= (1 .+ ϵ*Dxv).^2 + μ*(ϵ*Dz).^2 # Jacobien
			M1 = real.(-1im/sqrt(μ)*ifft(Π⅔ .* tanh.(ξ).* fft(Dphi)))
			M2 = real.( 1im*sqrt(μ)*ifft(Π⅔ .* cotanh(ξ) .* fft(M1./J )))
			q0 = mean((1 .+ ϵ*Dxv).*M2 + ϵ*μ*Dz.*M1./J)

			U1 .= (1 .+ ϵ*Dxv).*M1./J - ϵ*Dz.*M2 + ϵ*q0*Dz

		end
		function f2!(U1,U2)
		   	z .= U1
		   	Dphi .= U2
			ξ .= sqrt(μ)*(1 .+ ϵ*mean(z))*k
			#xv .=real.(ifft( Π⅔ .* cotanh(ξ) .* fft( z )))
			Dxv .= real.(sqrt(μ)*ifft(-1im* Π⅔ .* ∂ₓ.* cotanh(ξ) .* fft(z)))
			Dz .= real.(ifft(Π⅔ .* ∂ₓ.*fft(z)))

			J .= (1 .+ ϵ*Dxv).^2 + μ*(ϵ*Dz).^2 # Jacobien
			M1 = real.(-1im/sqrt(μ)*ifft(Π⅔ .* tanh.(ξ).* fft(Dphi)))
			M2 = real.( 1im*sqrt(μ)*ifft(Π⅔ .* cotanh(ξ) .* fft(M1./J )))
			q0 = mean((1 .+ ϵ*Dxv).*M2 + ϵ*μ*Dz.*M1./J)

			U2 .= real.(-ifft(∂ₓ .* fft( z +ϵ*Dphi.*M2 + ϵ/2 .*(Dphi.^2 -μ*M1.^2)./J - ϵ*q0*Dphi)))

		end


		new(label, f!, f1!, f2!, mapto, mapfro, param, kwargs )
    end
end
