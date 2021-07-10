export create_animation,plot_solution!,plot_solution
using Plots;
using ProgressMeter
"""
	create_animation( problems; name::String, kwargs... )

Creates an animation showing the evolution of initial-value problems.

Argument `problems` is either an element or a collection (vector, list, etc.) of elements of type `Problem`.

The animation is saved as `name.gif` if `name` is provided.

Other keyword arguments are as follows
- `xlims` allows to specifies the x axis limits for the surface deformation. If `nothing` is provided (default), then the full numerical basin is represented.
- `ylims` allows to specifies the y axis limits for the surface deformation. If `nothing` is provided (default), then the limits are determined from the initial data. If anything but a `Tuple` is provided, the axis limits evolve with the solution.
- `vlims` and `flims` are as above, but for the velocity and Fourier coefficients plots.
- `Nframes` gives the (maximal) number of frames in the animation.
- other arguments of `plot_solution!`

Return `anim` an animation, which can generate for instance a `gif` through `gif(anim, "my_name.gif", fps=15)`.

"""
function create_animation( problems; name=nothing, x = nothing, Nframes = 201,
							xlims=nothing, ylims=nothing,vlims=nothing,flims=nothing,
							interpol=false, surface=true,fourier=true, velocity=false )
	label=nothing
	if isa(problems,Problem) # if create_animation is called with a single problem
		problems=[problems];  # A one-element array to allow `for pb in p`
		label=problems[1].model.label; # saves the model's name
		problems[1].model.label=""; 	# replace it with blank so that it does not appear in the plots
	end

	if ylims == nothing
		(η,v) = solution(problems[1];t=0)
		y0 = minimum(η);y1 = maximum(η);
		ylims=(y0-(y1-y0)/10,y1+(y1-y0)/10)
	end
	if vlims == nothing
		(η,v) = solution(problems[1];t=0)
		y0 = minimum(v);y1 = maximum(v);
		vlims=(y0-(y1-y0)/10,y1+(y1-y0)/10)
	end
	if flims == nothing
		(η,v) = solution(problems[1];t=0)
		f=(abs.(fft(η)))
		y0 = minimum(f);y1 = maximum(f);
		flims=(y0-(y1-y0)/10,y1+(y1-y0)/10)
	end


	Ns=problems[1].times.Ns
	L=range(1,stop=Ns,step=max(1,ceil(Int,Ns/Nframes)))
	prog = Progress(length(L);dt=1,desc="Creating animation: ")
	m = surface + fourier + velocity
    anim = @animate for l in L
		plt = plot(layout=(m,1))
		for pb in problems
			plot_solution!( plt, pb; t=pb.times.ts[l],x=x,
					interpol=interpol,surface=surface,fourier=fourier, velocity=velocity )
		end
		n=1
		if surface
			if isa(ylims,Tuple) ylims!(plt[n,1],ylims) end
			if isa(xlims,Tuple) xlims!(plt[n,1],xlims) end
			n+=1
		end
		if velocity
			if isa(vlims,Tuple) ylims!(plt[n,1],vlims) end
			n+=1
		end
		if fourier
			if isa(flims,Tuple) ylims!(plt[n,1],flims) end
			n+=1
		end
        next!(prog)
    end
	if label!=nothing problems[1].model.label=label end # puts back the model's name
	if name != nothing gif(anim, string(name,".gif"), fps=15); end
	return anim
end

"""
	plot_solution!( plt; problems; t,x,interpol,surface,velocity,fourier )

Plots in `plt` the solution of initial-value problems at a given time.

# Argument
`problems` is either an element or a collection (vector, list, etc.) of elements of type `Problem`.

## Keywords
- `t` is the time. If not provided, then the last computed time is plotted.
- if a vector `x` is provided and if possible, the solution is interpolated to the collocation points `x`.
- if `interpol` is provided as an integer, the solution is interpolated on as many collocation points (if `true`, then the default value `2^3` is chosen).
- `surface`, `velocity` and `fourier` (booleans) determine respectively whether surface deformation, `η`, tangential velocity, `v`, and the Fourier coefficients of `η` (in log-scale) are plotted.

"""
function plot_solution!( plt, problems; t=nothing,x=nothing,interpol=false,surface=true, fourier=true, velocity=false, label=nothing )
	if typeof(problems)==Problem problems=[problems]; label = [label] end
	for i in 1:length(problems)
		p = problems[i]
		if label == nothing || label == [nothing]
			lbl=p.model.label
		else
			lbl=label[i]
		end
		if x != nothing
			(η,v) = solution(p)
			fftη = fft(η)
			fftv = fft(v)
			(η,v,X,t) = solution(p;t=t,x=x)
		else
			(η,v,X,t) = solution(p;t=t,interpol=interpol)
			fftη = fft(η)
			fftv = fft(v)
		end

		n=1
		if surface
	    	plot!(plt[n,1], X, η;
			  title=string("surface deformation at t=",t),
		      label=lbl)
			n+=1
		end

	  	if velocity
	    	plot!(plt[n,1], X, v;
			  title=string("tangential velocity at t=",t),
		      label=lbl)
			n+=1
		end

		if fourier
			if !all(isnan,fftη)
	    		plot!(plt[n,1], fftshift(p.mesh.k),
	          		eps(1.).+abs.(fftshift(fftη));
			  		title="Fourier coefficients (log scale)",
			  		#yaxis="log scale",
			  		yscale=:log10,
	    	  		label=lbl)
			end
		end
	end
	nothing
end

"""
	plot_solution( problems; t,x, surface,velocity,fourier )

Same as `plot_solution!` but generates and returns the plot.
"""
function plot_solution( problems; t=nothing,x=nothing, interpol=false, surface=true, velocity=false, fourier=true, label=nothing)
	plt = plot(layout=(surface+fourier+velocity,1))
	plot_solution!( plt, problems; t=t,x=x, interpol=interpol, surface=surface,fourier=fourier, velocity=velocity, label=label )
	return plt
end

function norm_problem!( plt, p::Problem, s::Real )
	N=[];
	Λ = sqrt.((p.mesh.k.^2).+1);
	prog = Progress(div(p.times.Ns,10),1)
 	for index in range(1,stop=p.times.Ns)
    	(hr,ur) = mapfro(p.model,p.data.U[index])
		push!(N,norm(ifft(Λ.^s.*fft(hr))))
		next!(prog)
	end

    plot!(plt, p.times.ts, N;
		  title=string("norm H^s avec s=",s),
	          label=p.model.label)

end
