export SolitaryWaveWhitham
using LinearMaps,IterativeSolvers,LinearAlgebra

"""
    `SolitaryWaveWhitham(param; kwargs...)`

Computes the Whitham solitary wave with prescribed velocity.

# Argument
- `param :: NamedTuple`: parameters of the problem containing velocity `c` and dimensionless parameters `ϵ` and `μ`, and mesh size `L` and number of collocation points `N`;

## Keywords
- `guess :: Vector{Real}`: initial guess for the surface deformation (if not provided, the exact formula for KdV is used);
- `x₀ :: Real`: center of solitary wave (if guess is not provided);
- `iterative :: Bool`: inverts Jacobian through GMRES if `true`, LU decomposition if `false`;
- `verbose :: Bool`: prints numerical errors at each step if `true`;
- `max_iter :: Int`: maximum number of iterations of the Newton algorithm;
- `tol :: Real`: general tolerance (default is `1e-10`);
- `ktol :: Real`: tolerance of the Krasny filter (default is `0`, i.e. no filtering);
- `gtol :: Real`: relative tolerance of the GMRES algorithm;
- `dealias :: Int`: dealiasing with Orlicz rule `1-dealias/(dealias+2)` (default is `0`, i.e. no dealiasing);
- `q :: Real`: Newton algorithm modified with
`u_{n+1}=q*u_{n+1}+(1-q)*u_n`
(default is `1`);
- `α :: Real`: adds `α` times spectral projection onto the Kernel to the Jacobian;
- `KdV :: Bool`: if `true` computes the KdV (instead of Whitham) solitary wave.
# Return values
`u :: Vector{Float64}` the solution
"""
function SolitaryWaveWhitham(
                param :: NamedTuple;
                guess = zeros(0) :: Vector{Float64},
                x₀ = 0 :: Real,
                iterative = false :: Bool,
                verbose = true :: Bool,
                max_iter = 20 :: Int,
                tol = 1e-14 :: Real,
                gtol = 1e-14 :: Real,
                ktol = 0 :: Real,
                dealias = 0 :: Int,
                q=1 :: Real,
                α=0 :: Real,
                KdV = false :: Bool)


        mesh=Mesh(param)
        c = param.c
        ϵ = param.ϵ
        μ = param.μ

        if guess == zeros(0)
                @info "Using the exact formula for the KdV solitary wave as initial guess"
                guess = 2*(c-1)/ϵ*sech.(sqrt(3*2*(c-1)/μ)/2*mesh.x.-x₀).^2
        end

        k  = mesh.k
        Dx =  1im * k
        if KdV == true
                @info string("Solving the KdV solitary wave with velocity c=",param.c)
                F₁ = 1 .-μ/6*k.^2
        else
                @info string("Solving the Whitham solitary wave with velocity c=",param.c)
                F₁ = sqrt.(tanh.(sqrt(μ)*abs.(k))./(sqrt(μ)*abs.(k)))
                F₁[1]=1
        end

        if dealias == 0
                Π = ones(size(k))
        else
                Π = abs.(k) .< maximum(k) * (1-dealias/(dealias+2))
        end
        krasny(k) = (abs.(k).> ktol ).*k
        krasny!(k) = k[abs.(k).< ktol ].=0

        function filter( v :: Vector{Float64} )
                real.(ifft(krasny(Π.*fft(v))))
        end
        function filter( v :: Vector{Complex{Float64}} )
                ifft(krasny(Π.*fft(v)))
        end


        function F( u :: Vector{Float64} )
            -c*u+real.(ifft(F₁.*fft(u)))+3*ϵ/4*u.^2
        end

        if iterative == false
                k = mesh.k
                x = mesh.x
                x₀ = mesh.x[1]
                FFT = exp.(-1im*k*(x.-x₀)');
                IFFT = exp.(1im*k*(x.-x₀)')/length(x);
                M = real.(IFFT*(Diagonal( F₁)*FFT))
                function JacF( v , dxv  )
                    Symmetric(M+Diagonal(3*ϵ/2*v .-c)+ α*dxv*dxv')
                end
        else
                function JacFfast( v ,dxv )
                        dF(φ) = -c*φ+real.(ifft(F₁.*fft(φ)))+3*ϵ/2*v.*φ+ α*dot(dxv,φ)*dxv
                        return LinearMap(dF, length(v); issymmetric=true, ismutating=false)
                end
        end

        flag=0
        iter = 0
        err = 1
        u = filter(guess)
        du = similar(u)
        fu = similar(u)
        dxu = similar(u)
        for i in range(1, length=max_iter)
                dxu .= real.(ifft(Dx.*fft(u)))
                dxu ./= norm(dxu,2)
                fu .= F(u)
    	        err = norm(fu,Inf)
    		if err < tol
    			@info string("Converged : ",err,"\n")
    			break
    		elseif verbose == true
                        print(string("error at step ",i,": ",err,"\n"))
    		end
                if i == max_iter
                        @warn  string("The algorithm did not converge: ",err,"\n")
                end
                if iterative == false
                        du .= JacF(u,dxu) \ fu
                else
                        du .= gmres( JacFfast(u,dxu) , fu ; verbose = verbose, tol = gtol )
                end
    		u .-= q*filter(du)
        end
        return u
end
