export CGBSW
using FFTW
"""
    CGBSW( params )

"""
mutable struct CGBSW <: AbstractModel

    label   :: String
	f!		:: Function
	mapto	:: Function
	mapfro	:: Function

    function CGBSW( param::NamedTuple)

        label = "Cheng et al."
        datasize = 2
        ϵ = param.ϵ
        mesh  = Mesh(param)
        x = mesh.x
        Γ = abs.(mesh.k)
        Dx    =  1im * mesh.k            # Differentiation
        H     = -1im * sign.(mesh.k)     # Hilbert transform
        Π⅔    = Γ .< mesh.kmax * 2/3 # Dealiasing low-pass filter

        hnew = zeros(Complex{Float64}, mesh.N)
        unew = zeros(Complex{Float64}, mesh.N)

        I₁ = zeros(Complex{Float64}, mesh.N)
        I₂ = zeros(Complex{Float64}, mesh.N)
        I₃ = zeros(Complex{Float64}, mesh.N)

        Px  = plan_fft(hnew; flags = FFTW.MEASURE)

		function f!(U)

		    ldiv!(hnew, Px , view(U,:,1))

		    I₁  .= view(U,:,2) .* Γ
		    ldiv!(unew, Px , I₁)
		    unew .^= 2
		    mul!(I₁, Px , unew)
		    I₁ .*= H

		    I₂  .= view(U,:,1) .* Dx
		    ldiv!(unew, Px , I₂)
		    unew .*= hnew
		    mul!(I₂, Px , unew)

		    I₃  .= view(U,:,1) .* Γ
		    ldiv!(unew, Px, I₃)
		    unew .*= hnew
		    mul!(I₃ , Px , unew)
		    I₃ .*= H

		    hnew  .= .- view(U,:,2) .* Dx

		    I₁ .-= I₂
		    I₁ .-= I₃
		    I₁ .*= Π⅔
		    I₁ .*= ϵ

		    U[:,2] .= view(U,:,1) .* H .+ I₁
		    U[:,1] .= hnew

		end

		"""
		    mapto(CGBSW, data)
		    the velocity should be zero

		"""
		function mapto(data::InitialData)
		    [Π⅔ .* fft(data.η(x)) Π⅔ .*fft(data.v(x))]
		end

		function mapfro(U)
			real(ifft(U[:,1])),real(ifft(U[:,2]))
		end

		new(label, f!, mapto, mapfro )

    end
end
