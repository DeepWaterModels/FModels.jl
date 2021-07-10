export CnoidalWaveWhithamGreenNaghdi
using Elliptic
using LinearMaps,IterativeSolvers,LinearAlgebra

"""
    `CnoidalWaveWhithamGreenNaghdi(param; kwargs...)`

Computes the Serre-Green-Naghdi cnoidal wave with prescribed `h₀<h₁<h₂`.
`h_1` is the minimum, `h_2` is the maximum.
As `h₀ -> h₁`, the cnoidal wave converges towards the solitary wave.

# Arguments
- `param :: NamedTuple`: parameters of the problem containing `h₀<h₁<h₂` and dimensionless parameters `ϵ` and `μ`, and number of collocation points `N`.
## Keywords
- `guess :: Vector{Float64}`: initial guess for the surface deformation (if not provided, the exact formula for SGN is used);
- `P :: Int`: the number of periods of the cnoidal wave in the constructed mesh;
- `SGN :: Bool`: if `true` computes the Serre-Green-Naghdi (instead of Whitham-Green-Naghdi) solitary wave;
- `method :: Int`: equation used (between `1` and `4`);
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
- `α :: Real`: adds `α` times spectral projection onto the Kernel to the Jacobian (default is `0`).
# Return values
`(η,u,v,mesh,param)` with
- `η :: Vector{Float64}`: surface deformation;
- `u :: Vector{Float64}`: layer-averaged velocity;
- `v :: Vector{Float64}`: tangential velocity;
- `mesh :: Vector{Float64}` collocation points;
- `param :: NamedTuple`: useful parameters
"""
function CnoidalWaveWhithamGreenNaghdi(
                param :: NamedTuple;
                guess = zeros(0) :: Vector{Float64},
                P = 2 :: Int,
                SGN = false :: Bool,
                method = 2 :: Int,
                iterative = false :: Bool,
                verbose = true :: Bool,
                max_iter = 0 :: Int,
                tol = 1e-10 :: Real,
                ktol = 0 :: Real,
                gtol = 1e-10 :: Real,
                dealias = 0 :: Int,
                q=1 :: Real,
                α=0 :: Real
                        )
        if SGN == true
                @info string("Computing the SGN cnoidal wave with parameters",(param.h₀,param.h₁,param.h₂))
        else
                @info string("Computing the WGN solitary wave with parameters",(param.h₀,param.h₁,param.h₂))
                error("Computing the WGN solitary wave has not been coded yet. Sorry.")
        end

        ϵ = param.ϵ
        μ = param.μ

        if guess == zeros(0)
                @info "Using the exact formula for the SGN solitary wave as initial guess"
                # a₁ = ϵ*param.a₁
                # m = param.m
                # a₀= 1-a₁*Elliptic.E(m)/Elliptic.K(m)
                # c = sqrt(a₀ * (a₀+a₁) * (a₀+(1-m^2)*a₁) )
                # κ = sqrt(3*a₁)/(2*c)/sqrt(μ)
                # #guess = a₀ .-1 .+ a₁*(Jacobi.dn.(κ*mesh.x,m).^2)
                # h₀ = a₀
                # h₁ = a₀+(1-m^2)*a₁
                # h₂ = a₀+a₁
                h₀=param.h₀
                h₁=param.h₁
                h₂=param.h₂
                c = sqrt(h₀*h₁*h₂)
                @info string("The velocity is c=",c)
                m = sqrt((h₂-h₁)/(h₂-h₀))
                κ = sqrt(3*(h₂-h₀))/(2*c)/sqrt(μ)
                λ = Elliptic.K(m^2)/κ
                @info string("The period is 2*λ=",2*λ)
                mesh = Mesh((L=P*λ,N=param.N))
                guess = h₁ .-1 .+ (h₂-h₁)*(Jacobi.cn.(κ*mesh.x,m^2).^2)

                a₀ = h₀
                a₁ = h₂ - h₀
                guess2 = a₀ .-1 .+ a₁*(Jacobi.dn.(κ*mesh.x,m^2).^2)
                H₀= a₀+a₁*Elliptic.E(m^2)/Elliptic.K(m^2)
                u2 = c*(1 ./ H₀ .- 1 ./(1 .+ guess2))
                param = (h₀=h₀,h₁=h₁,h₂=h₂,a₀=a₀,a₁=a₁,H₀=H₀,c=c,λ=λ,m=m,κ=κ)
                if SGN == true
                        max_iter = 0
                end
        else
                c = param.c
                mesh=Mesh((L=param.L,N=param.N))
        end

        k = mesh.k

        Dx       =  1im * k
        if SGN == true
                F₀ = sqrt(μ)*Dx #./ (1 .+ μ/3 * k.^2 ).^(1/4)
        else
                F₁ 	= tanh.(sqrt(μ)*abs.(k))./(sqrt(μ)*abs.(k))
                F₁[1] 	= 1
                F₀       = 1im * sqrt.(3*(1 ./F₁ .- 1)).*sign.(k)
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
        function F( v , hv, Fv, F2v )
                if method == 1
                        return -1/3 ./ hv.^2 .* F2v .+
                                ϵ/2 .* (hv.*Fv).^2 .+
                                v./hv .- 1/(c^2) .*v .- ϵ/2 .* (v./hv).^2
                elseif method == 2
                        return -c/3 ./ hv.^2 .* F2v .+
                                ϵ/2 .* Fv.^2 .+
                                c*v .- 1/c .*v.*hv .- ϵ/2 .* v.^2
                elseif method == 3
                        return -1/3 ./ hv.^2 .* F2v .+
                                ϵ/2 .* Fv.^2 .+
                                v .- 1/(c^2) .*v.*hv .- ϵ/2 .* v.^2
                elseif method == 4
                        return (-1/3 .* F2v .+
                                ϵ/2 .*  hv.^4 .* Fv.^2 .+
                                hv.^2 .* (v .- ϵ/2 .* v.^2) .-
                                1/(c^2) .* hv.^3 .* v)/c^4
                end
        end


        function Fabs( v , hv, Fv, F2v )
                if method == 1
                        return abs.(1/3 ./ hv.^2 .* F2v) .+
                                abs.(ϵ/2 .* (hv.*Fv).^2 ) .+
                                abs.(v./hv) .- 1/(c^2) .*abs.(v) .+ ϵ/2 .* abs.((v./hv).^2)
                elseif method == 2
                        return abs.(c/3 ./ (hv.^2).* F2v) .+
                                abs.(ϵ/2 .* Fv.^2) .+
                                abs.(c*v) .+ abs.(1/c .*v.*hv) .+ abs.(ϵ/2 .* v.^2)
                elseif method == 3
                        return abs.(1/3 ./ (hv.^2).* F2v) .+
                                abs.(ϵ/2 .* Fv.^2) .+
                                abs.(v) .+ abs.(1/(c^2) .*v.*hv) .+ abs.(ϵ/2 .* v.^2)
                elseif method == 4
                        return (abs.(-1/3 .* F2v) .+
                                abs.(ϵ/2 .*  hv.^4 .* Fv.^2) .+
                                abs.( hv.^2 .*v) .+ abs.( ϵ/2 .* hv.^2 .* v.^2) .+
                                1/(c^2) .* abs.(hv.^3 .* v))/c^4
                end
        end
        if iterative == false
                x = mesh.x
                x₀ = mesh.x[1]
                FFT = exp.(-1im*k*(x.-x₀)');
                IFFT = exp.(1im*k*(x.-x₀)')/length(x);
                #Id = Diagonal(ones(size(x)));
                M₀ = IFFT * Diagonal( F₀ )* FFT
                M(v) = Diagonal( v )
                function JacF( v , hv, Fv , F2v, Dv, dxv )
                        if method == 1
                                return Symmetric(real.(
                                        -1/3 *M(1 ./ hv.^2)* M₀ * M(hv.^3)* ( M₀ * M( 1 ./ hv.^2 ) .+ 3* ϵ * M( Fv ./hv ) )
                                        .+ ϵ * M( hv.^2 .* Fv ) * M₀ * M( 1 ./ hv.^2 )
                                        .+ M( Dv ) .+ α*dxv*dxv' ))
                        elseif method == 2
                                return real.(
                                        -1/3 *M(1 ./ hv.^2 )* M₀ * M(hv.^3)* (c*M₀ .+ 3*ϵ * M( Fv ))
                                        .+ ϵ * M( hv .* Fv ) * M₀
                                        #.+ M(1 ./hv) .+ 2*ϵ/3 * M(F2v ./ hv) .+ ϵ^2 * M(hv .* Fv.^2) .- M((hv/c).^2)
                                        .+ M( Dv ) .+ α*M( hv.^2 )*dxv*dxv' *M(1 ./ hv.^2 ) )
                        elseif method == 3
                                return real.(
                                        -1/3 *M(1 ./ hv.^2 )* M₀ * M(hv.^3)* (M₀ .+ 3*ϵ * M( Fv ))
                                        .+ ϵ * M( hv .* Fv ) * M₀
                                        #.+ M(1 ./hv) .+ 2*ϵ/3 * M(F2v ./ hv) .+ ϵ^2 * M(hv .* Fv.^2) .- M((hv/c).^2)
                                        .+ M( Dv ) .+ α*M( hv.^2 )*dxv*dxv' *M(1 ./ hv.^2 ) )
                        elseif method == 4
                                return Symmetric(real.(
                                ( M₀ * M(hv.^3 ) * ( -1/6*M₀ .- ϵ * M( hv .* Fv ) ) .+
                                 ( -1/6*M₀ .+ ϵ * M( hv .* Fv ) ) * M( hv.^3  ) * M₀ .+
                                 M(hv.^2)*M( Dv ))/c^4 .+ α*dxv*dxv'
                                ))
                        end
                end
        else
                Four(v) = (ifft(F₀.*fft(v)))
                function JacFfast( v , hv, Fv , F2v, Dv, dxv )
                        function dF(φ)
                                if method == 1
                                 return fft( -1/3 ./ hv.^2 .* Four( hv.^3 .*
                                                ( Four( ifft(φ) ./ hv.^2 ) .+ 3* ϵ * ifft(φ) .* Fv ./hv )  )
                                        .+ ϵ .* hv.^2 .* Fv .* Four( ifft(φ) ./ hv.^2 )
                                         .+ ifft(φ) .* Dv .+ α*dot(dxv,ifft(φ))*dxv )
                                 elseif method == 2
                                  return fft( -1/3 ./ hv.^2 .* Four( hv.^3 .* ( c*Four( ifft(φ) ) .+ 3*ϵ*Fv.*ifft(φ) ))
                                          .+ ϵ .* hv .* Fv .* Four( ifft(φ) )
                                          .+ ifft(φ) .* Dv .+ α*dot(dxv, ifft(φ) ./ hv.^2)*hv.^2 .*dxv )
                                  elseif method == 3
                                   return fft( -1/3 ./ hv.^2 .* Four( hv.^3 .* ( Four( ifft(φ) ) .+ 3*ϵ*Fv.*ifft(φ) ))
                                           .+ ϵ .* hv .* Fv .* Four( ifft(φ) )
                                           .+ ifft(φ) .* Dv .+ α*dot(dxv, ifft(φ) ./ hv.^2)*hv.^2 .*dxv )
                                elseif method == 4
                                 return fft(( -1/3  .* Four( hv.^3 .* ( Four( ifft(φ) ) .+ 3*ϵ*Fv.*hv.*ifft(φ) ))
                                        .+ ϵ .* hv.^4 .* Fv .* Four( ifft(φ) )
                                        .+ ifft(φ) .* hv.^2 .* Dv)/c^4 .+ α*dot(dxv,ifft(φ))*dxv)
                                end
                         end
                if method == 1 || method == 4
                        return LinearMap(dF, length(v); issymmetric=true, ismutating=false)
                else
                        return LinearMap(dF, length(v); issymmetric=false, ismutating=false)
                end
                end

        end
#------ Iteration scheme

        # initial guess for iteration
        if method == 1
                u = filter(guess)
        elseif method == 2
                u = c*filter(guess./(1 .+ ϵ*guess))
        elseif method == 3 || method == 4
                u = filter(guess./(1 .+ ϵ*guess))
        end

        du,fu,dxu,hu,Fu,F2u,Du = (similar(u),).*ones(7)

        for i in range(1, length=max_iter)
                dxu .= real.(ifft(Dx.*fft(u)))
                dxu ./= norm(dxu,2)
                if method == 1
                        hu .= 1 .+ϵ*u
                        Fu .= real.(ifft(F₀.*fft(u./hu)))
                        F2u .= real.(ifft(F₀.*fft( hu.^3 .* Fu ) ))
                        Du .= 1 ./ hu.^3 .+ 2*ϵ/3 ./ hu.^3 .* F2u .+ ϵ^2 * hu.* Fu.^2 .- 1/c^2
                elseif method == 2
                        hu .= c ./(c .- ϵ*u)
                        Fu .= hu.* real.(ifft(F₀.*fft(u)))
                        F2u .= real.(ifft(F₀.*fft(hu.^2 .* Fu ) ))
                        Du .= c ./hu .+ 2*ϵ/3 * F2u ./ hu .+ ϵ^2/c * hu .* Fu.^2 .- (hu.^2)/c
                elseif method == 3
                        hu .= 1 ./(1 .- ϵ*u)
                        Fu .= hu.* real.(ifft(F₀.*fft(u)))
                        F2u .= real.(ifft(F₀.*fft(hu.^2 .* Fu ) ))
                        Du .= 1 ./hu .+ 2*ϵ/3 * F2u ./ hu .+ ϵ^2 * hu .* Fu.^2 .- (hu/c).^2
                elseif method == 4
                        hu .= 1 ./(1 .- ϵ*u)
                        Fu .= real.(ifft(F₀.*fft(u)))
                        F2u .= real.(ifft(F₀.*fft(hu.^3 .* Fu ) ))
                        Du .=  1 ./ hu .+ 2*ϵ^2 * hu.^3 .* Fu.^2  .+ 2*ϵ.* hu .*(u.- ϵ/2* u.^2).-(hu .+ 3*ϵ* u.* hu.^2 )./(c^2)
                end
                fu .= F(u,hu,Fu,F2u)

                err = norm(fu,Inf)#/norm(Fabs(u,hu,Fu,F2u),Inf)
    		if err < tol
    			@info string("Converged : ",err,"\n")
    			break
    		elseif verbose == true
                        print(string("error at step ",i,": ",err,"\n"))
    		end
                if i == max_iter
                        @warn  string("The algorithm did not converge: final error is ",err,"\n")
                end

                if iterative == false
                        du .=  JacF(u,hu,Fu,F2u,Du,dxu)  \ fu
                else
                        x = mesh.x
                        x₀ = mesh.x[1]
                        FFT = exp.(-1im*k*(x.-x₀)');
                        IFFT = exp.(1im*k*(x.-x₀)')/length(x);
                        F₁ 	= tanh.(sqrt(μ)*abs.(k))./(sqrt(μ)*abs.(k))
                        F₁[1] 	= 1
                        F₁ = 1 ./ (1 .+ μ*k.^2   )
                        #F₁ = ones(length(k))
                        Precond = Diagonal( 1 ./ F₁  )
                        du .=  real.(ifft(gmres( JacFfast(Complex.(u),hu,Fu,F2u,Du,dxu) , fft(fu) ; Pl = Precond, tol = gtol, verbose=verbose )))
                end
    		u .-= q*filter(du)
        end

        # if method == 1
        #         return (mesh,guess,u,c*u./(1 .+ϵ*u))
        # elseif method ==2
        #         return (mesh,guess,u./(c .- ϵ*u),u)
        # elseif method ==3 || method == 4
        #         return (mesh,guess,u./(1 .- ϵ*u),c*u)
        # end
        η = guess/ϵ
        h = 1 .+ ϵ*η
        u = c*η./h
        DxF(v) = real.(ifft(F₀ .* fft(v)))
	v = u - 1/3 ./h .* (DxF(h.^3 .*DxF(u)))
        #
        # h2 = 1 .+ ϵ*guess2
        # v2 = u2 - 1/3 ./h2 .* (DxF(h2.^3 .*DxF(u2)))


        return (η,u,v,mesh,param)

end
