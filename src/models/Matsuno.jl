export Matsuno
using FFTW,LinearAlgebra

"""
    Matsuno(params)

"""
mutable struct Matsuno <: AbstractModel

    label   :: String
	f!		:: Function
	mapto	:: Function
	mapfro	:: Function

    function Matsuno(param::NamedTuple)

        label    = "Matsuno"
        ϵ        = param.ϵ
        mesh     = Mesh(param)
        x        = mesh.x
        Γ        = abs.(mesh.k)
        Dx       =  1im * mesh.k        # Differentiation
        H        = -1im * sign.(mesh.k) # Hilbert transform
        Π⅔       = Γ .< mesh.kmax * 2/3 # Dealiasing low-pass filter

        hnew = zeros(Complex{Float64}, mesh.N)
        unew = zeros(Complex{Float64}, mesh.N)

        I₀ = zeros(Complex{Float64}, mesh.N)
        I₁ = zeros(Complex{Float64}, mesh.N)
        I₂ = zeros(Complex{Float64}, mesh.N)
        I₃ = zeros(Complex{Float64}, mesh.N)

        Px  = plan_fft(hnew; flags = FFTW.MEASURE)

		function f!(U)

		    for i in eachindex(hnew)
		        hnew[i] = Γ[i] * U[i,1]
		    end

		    ldiv!(unew, Px, hnew )

		    for i in eachindex(hnew)
		        hnew[i] = Dx[i] * U[i,1]
		    end

		    ldiv!(I₁, Px, hnew)

		    unew  .*= I₁

		    mul!(I₁, Px, unew)

		    I₁  .*= ϵ .* Π⅔
		    I₁  .-= hnew

		    ldiv!(hnew, Px, view(U,:,1))
		    ldiv!(unew, Px, view(U,:,2))

		    I₂    .= hnew .* unew

		    mul!(I₃, Px, I₂)

		    I₃    .*= Dx

		    for i in eachindex(H)
		        U[i,1]  = H[i] * U[i,2]
		        I₀[i] = Γ[i] * U[i,2]
		    end

		    ldiv!(I₂, Px, I₀)

		    I₂    .*= hnew

		    mul!(hnew, Px, I₂)

		    hnew  .*= H
		    I₃    .+= hnew
		    I₃    .*= ϵ .* Π⅔

		    for i in eachindex(I₃)
		        U[i,1] -= I₃[i]
		    end

		    I₃    .=  unew.^2

		    mul!(unew, Px, I₃)

		    unew  .*= Dx
		    unew  .*= ϵ/2 .* Π⅔
		    I₁    .-= unew

		    for i in eachindex(I₁)
		        U[i,2] =  I₁[i]
		    end

		end

		"""
		    mapto(Matsuno, data)
		    the velocity should be zero

		"""
		function mapto(data::InitialData)

		    [Π⅔ .* fft(data.η(x)) Π⅔ .*fft(data.v(x))]

		end

		function mapfro(U)
		    real(ifft(view(U,:,1))),real(ifft(view(U,:,2)))
		end

		new(label, f!, mapto, mapfro )
    end
end
