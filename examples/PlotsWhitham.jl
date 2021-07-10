# # Compute the solitary wave of the Whitham equation with prescribed velocity
@info "Defines functions PlotSolitaryWaveKdV,PlotSolitaryWaveWhitham"
using ShallowWaterModels,FFTW,Plots;
include("../src/initialdata/SolitaryWaveWhitham.jl")

#---- KdV
"""
	`PlotSolitaryWaveKdV()`
A proof of concept: numerically computes the solitary wave of the KdV equation with prescribed velocity c=5
"""
function PlotSolitaryWaveKdV()
	param = ( μ  = 1,
			ϵ  = 1,
        	N  = 2^10,
            L  = 10,
			)
	mesh = Mesh(param)

	function sol(x,α)
		2*α*sech.(sqrt(3*2*α)/2*x).^2
	end

	u₀ = SolitaryWaveWhitham( merge(param,(c=2.5,)); guess = sol(mesh.x,1.5)+.1*exp.(-(mesh.x).^2), iterative = false, KdV = true, α=1,tol = 1e-8)
	u₁ = SolitaryWaveWhitham( merge(param,(c=2.6,)); guess = sol(mesh.x,1.6)+.1*exp.(-(mesh.x).^2), iterative = false, KdV = true, α=1,tol = 1e-8)
	u₂ = SolitaryWaveWhitham( merge(param,(c=2.7,)); guess = sol(mesh.x,1.7)+.5*exp.(-(mesh.x).^2), iterative = false, KdV = true, α=1,tol = 1e-8)

	# barycentric Lagrange interpolation (4.2) in https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
	function P(X,dc,u₀,u₁,u₂)
		(u₀/(2*(X/dc+2))-u₁/(X/dc+1)+u₂/(2*X/dc))/(1/(2*(X/dc+2))-1/(X/dc+1)+1/(2*X/dc))
	end


	function solve!(u₀,u₁,u₂,c,dc)
			tu₂ = copy(u₂)
			u₂ .= SolitaryWaveWhitham( merge(param,(c=c,)); guess = P.(dc,dc,u₀,u₁,u₂), iterative = false, α=1, KdV = true, verbose = true, max_iter = 5, tol = 1e-10)
			u₀ .= u₁
			u₁ .= tu₂
	end

	for cs = range(2.8; step = 0.1, stop = 5)
		solve!(u₀,u₁,u₂,cs,0.1)
	end

	ucomp = u₂
	uexact = sol(mesh.x,4)

	plt = plot(layout=(2,2))

    plot!(plt[1,1], mesh.x, [ucomp uexact];
     title=string("c=5"),
     label=["computed" "exact"])

    plot!(plt[2,1], fftshift(mesh.k),
     [log10.(abs.(fftshift(fft(ucomp)))) log10.(abs.(fftshift(fft(uexact))))];
     title="frequency",
	 label=["computed" "exact"])

	 plot!(plt[1,2], mesh.x, ucomp-uexact;
      title=string("c=5"),
      label="difference")

     plot!(plt[2,2], fftshift(mesh.k),
      log10.(abs.(fftshift(fft(ucomp-uexact))));
      title="frequency",
	  label="difference")
end


#----
"""
	`PlotSolitaryWaveWhitham(c)`

Computes the solitary wave of the Whitham equation with prescribed velocity.

`c` is the velocity, and should be more between `1` and `1.2290408`.
"""
function PlotSolitaryWaveWhitham(c)
 function solKdV(x,α)
   	 2*α*sech.(sqrt(3*2*α)/2*x).^2
 end
 if c<=1 || c>1.2290408
	 error("The velocity should be between `1` and `1.2290408`")
 elseif c<1.15
	param = ( μ  = 1,
 			ϵ  = 1,
         	N  = 2^10,
             L  = 10/sqrt(c-1),
 			)
 	mesh = Mesh(param)
	u = SolitaryWaveWhitham( merge(param,(c=c,)); iterative = false)

 else
	param = ( μ  = 1,
			ϵ  = 1,
        	N  = 2^10,
            L  = 15,
			)
	mesh = Mesh(param)

	# barycentric Lagrange inerpolation (4.2) in https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
	function P(X,dc,u₀,u₁,u₂)
		(u₀/(2*(X/dc+2))-u₁/(X/dc+1)+u₂/(2*X/dc))/(1/(2*(X/dc+2))-1/(X/dc+1)+1/(2*X/dc))
	end
	#
	# function Q(X,dc,u₀,u₁,u₂)
	# 	(u₀./(2*(X/dc+2)) .-u₁./(X/dc+1) .+u₂./(2*X/dc))./(1/(2*(X/dc+2))-1/(X/dc+1)+1/(2*X/dc))
	# end

	function solve2!(u₀,u₁,u₂,c,dc)
			tu₂ = copy(u₂)
			u₂ .= SolitaryWaveWhitham(merge(param,(c=c,)); guess = P.(dc,dc,u₀,u₁,u₂), iterative = false, verbose = true, max_iter = 5)
			u₀ .= u₁
			u₁ .= tu₂
	end

	function solve2!(u₀,u₁,u₂,c,old_dc,new_dc)
			tu₀ = copy(u₀)
			tu₁ = copy(u₁)
			tu₂ = copy(u₂)
			u₀ .= tu₂
			u₁ .= SolitaryWaveWhitham( merge(param,(c=c+1*new_dc,)); guess = P.(1*new_dc,old_dc,tu₀,tu₁,tu₂), iterative = false, verbose = true)
			u₂ .= SolitaryWaveWhitham( merge(param,(c=c+2*new_dc,)); guess = P.(2*new_dc,old_dc,tu₀,tu₁,tu₂), iterative = false, verbose = true)
	end

	u₀ = SolitaryWaveWhitham( merge(param,(c=1.10,)); iterative = false)
	u₁ = SolitaryWaveWhitham( merge(param,(c=1.11,)); iterative = false)
	u₂ = SolitaryWaveWhitham( merge(param,(c=1.12,)); iterative = false)

	if c < 1.22
		for cs = range(1.13; step = 0.01, stop = c)
			solve2!(u₀,u₁,u₂,cs,0.01)
		end
	else
		for cs = range(1.13; step = 0.01, stop = 1.22)
			print(string("c = ",cs,"\n"))
			solve2!(u₀,u₁,u₂,cs,0.01)
		end

		solve2!(u₀,u₁,u₂,1.221,0.01,1e-4)

		if c < 1.229
			for cs = range(1.221+3e-4; step = 1e-4, stop = c)
				solve2!(u₀,u₁,u₂,cs,1e-4)
			end
		else
			c=1.2290408
			for cs = range(1.221+3e-4; step = 1e-4, stop = 1.229)
				solve2!(u₀,u₁,u₂,cs,1e-4)
			end

			solve2!(u₀,u₁,u₂,1.229,1e-4,1e-6)

			for cs = range(1.229+3e-6; step = 1e-6, stop = 1.22904)
				solve2!(u₀,u₁,u₂,cs,1e-6)
			end

			solve2!(u₀,u₁,u₂,1.22904,1e-6,1e-8)

			for cs = range(1.22904+3e-8; step = 1e-8, stop = 1.2290407)
				solve2!(u₀,u₁,u₂,cs,1e-8)
			end
			for cs = range(1.2290407+1e-8; step = 1e-8, stop = 1.2290408)
				solve2!(u₀,u₁,u₂,cs,1e-8)
			end
		end
	end
	u = u₂
 end
 plt = plot(layout=(1,2))
 plot!(plt[1,1], mesh.x, [u solKdV(mesh.x,c-1)];
  title=string("c=",c),
  label=["Whitham" "KdV"])

 plot!(plt[1,2], fftshift(mesh.k),
  log10.(abs.(fftshift(fft(u))));
  title="frequency",
  label="Whitham")
end
nothing
