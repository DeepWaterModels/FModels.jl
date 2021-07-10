export CGBSW_naive
using FFTW
"""
    CGBSW_naive( params )

"""
mutable struct CGBSW_naive <: AbstractModel

	label   :: String
	f!		:: Function
	mapto	:: Function
	mapfro	:: Function

    function CGBSW_naive( param::NamedTuple)

        label = "Cheng et al. naive"
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

		function f!(U)

			hnew .=ifft(U[:,1]);
			I₁ .=H.*fft(ifft(Γ.*U[:,2]).^2);
			I₂ .=fft(hnew.*ifft(Dx.*U[:,1])) ;
			I₃ .=H.*fft(hnew.*ifft(Γ.*U[:,1]));
			hnew .= U[:,1] ;
			U[:,1] .= -(Dx.*U[:,2]) ;
			U[:,2] .= H.*hnew+ϵ*Π⅔.*(I₁-I₂-I₃) ;

		end

		"""
		    mapto(CGBSW_naive, data)
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
