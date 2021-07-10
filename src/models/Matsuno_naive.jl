export Matsuno_naive
using FFTW
"""
    Matsuno(params)

"""
mutable struct Matsuno_naive <: AbstractModel

	label   :: String
	f!		:: Function
	mapto	:: Function
	mapfro	:: Function

    function Matsuno_naive(param::NamedTuple)

		label = "Matsuno naive"
		ϵ 	= param.ϵ
		mesh = Mesh(param)
		x   = mesh.x
        Γ 	= abs.(mesh.k)
    	∂ₓ	=  1im * mesh.k            # Differentiation
        H 	= -1im * sign.(mesh.k)     # Hilbert transform
        Π⅔ 	= Γ .< mesh.kmax * 2/3 # Dealiasing low-pass filter

        hnew = zeros(Complex{Float64}, mesh.N)
        unew = zeros(Complex{Float64}, mesh.N)

        I₁ = zeros(Complex{Float64}, mesh.N)
        I₂ = zeros(Complex{Float64}, mesh.N)
        I₃ = zeros(Complex{Float64}, mesh.N)


		function f!(U)

		   hnew .= ifft(U[:,1])
		   unew .= ifft(U[:,2])
		   I₃ .= fft(ifft((∂ₓ).*U[:,1]).*ifft((Γ).*U[:,1]))
		   I₁ .= H.*U[:,2].-ϵ*Π⅔.*(H.*fft(hnew.*ifft(Γ.*U[:,2])).+∂ₓ.*fft(hnew.*unew))
		   I₂ .= -(∂ₓ.*U[:,1])+ϵ*Π⅔.*(I₃-∂ₓ.*fft(unew.^2)/2)
		   #
		   U[:,1] .= I₁
		   U[:,2] .= I₂

		end

		"""
		    mapto(Matsuno, data)
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
