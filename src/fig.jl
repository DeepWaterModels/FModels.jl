export create_animation,fig_problem!,fig_problem

function create_animation( p::Problem;str="anim"::String )

	prog = Progress(p.times.Ns;dt=1,desc="Creating animation: ")
	(h0,u0) = mapfro(p.model,first(p.data.U))
	y0 = minimum(h0);y1 = maximum(h0);
    anim = @animate for l in range(1,stop=min(p.times.Ns-1,200))

        plt = plot(layout=(2,1))

		(hr,ur) = mapfro(p.model,p.data.U[l])

        plot!(plt[1,1], p.mesh.x, hr;
              title="physical space",
			  ylims=(y0-(y1-y0)/10,y1+(y1-y0)/10)
			  )

        plot!(plt[2,1], fftshift(p.mesh.k),
              log10.(1e-18.+abs.(fftshift(fft(hr))));
              title="frequency")

        next!(prog)

    end

    gif(anim, string(str,".gif"), fps=15); nothing

end


function create_animation( pbs::Array{Any,1};str="anim"::String )
	p0=pbs[1]
    prog = Progress(p0.times.Ns;dt=1,desc="Creating animation: ")

    anim = @animate for l in range(1,stop=min(p0.times.Ns-1,200))

        plt = plot(layout=(2,1))

		for p in pbs
			if typeof(p.model)==WaterWaves
				(x,z,v) = mapfro(p.model,p.data.U[l])
				plot!(plt[1,1], x, z;
					title="physical space",
					label=p.model.label)

				plot!(plt[2,1], fftshift(p.mesh.k),
					log10.(1e-18.+abs.(fftshift(fft(z .- 1))));
					title="frequency",
					label=p.model.label)


			else
				(hr,ur) = mapfro(p.model,p.data.U[l])

        		plot!(plt[1,1], p.mesh.x, hr;
              		title="physical space",
              		label=p.model.label)

        		plot!(plt[2,1], fftshift(p.mesh.k),
              		log10.(1e-18.+abs.(fftshift(fft(hr))));
              		title="frequency",
              		label=p.model.label)
			end
		end
        next!(prog)

    end

    gif(anim, string(str,".gif"), fps=15); nothing

end

#----
#
#
#md # Plot results function
function fig_problem!( plt, p::Problem )

	if typeof(p.model)==WaterWaves

		(x,z,v) = mapfro(p.model,p.data.U[end])
		plot!(plt[1,1], x, z;
			title="physical space",
			label=p.model.label)

		plot!(plt[2,1], fftshift(p.mesh.k),
			log10.(1e-18.+abs.(fftshift(fft(z .- 1))));
			title="frequency",
			label=p.model.label)

	else
    	(hr,ur) = mapfro(p.model,p.data.U[end])

    	plot!(plt[1,1], p.mesh.x, hr;
		  	title="physical space",
	        label=p.model.label)

    	plot!(plt[2,1], fftshift(p.mesh.k),
            log10.(1e-18.+abs.(fftshift(fft(hr))));
		  	title="frequency",
    	    label=p.model.label)
	end

end

function fig_problem!( plt, p::Problem, t::Real )

	t=max(t,0)
	t=min(t,p.times.tfin)
	index = indexin(false,p.times.ts.<t)[1]

	if typeof(p.model)==WaterWaves

		(x,z,v) = mapfro(p.model,p.data.U[index])
		plot!(plt[1,1], x, z;
			title="physical space",
			label=p.model.label)

		plot!(plt[2,1], fftshift(p.mesh.k),
			log10.(1e-18.+abs.(fftshift(fft(z .- 1))));
			title="frequency",
			label=p.model.label)

	else
    	(hr,ur) = mapfro(p.model,p.data.U[index])

    	plot!(plt[1,1], p.mesh.x, hr;
		  title=string("physical space at t=",p.times.ts[index]),
	          label=p.model.label)

    	plot!(plt[2,1], fftshift(p.mesh.k),
                  log10.(1e-18.+abs.(fftshift(fft(hr))));
		  title="frequency",
    	          label=p.model.label)
	end
end
function fig_problem( p::Problem, t::Real )
	plt = plot(layout=(2,1))
	fig_problem!( plt, p::Problem, t::Real )
	display(plt)
	return plt
end
function fig_problem( p::Problem)
	plt = plot(layout=(2,1))
	fig_problem!( plt, p::Problem )
	display(plt)
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


#
# function fig(t)
#     s=0
#
#     if indexin(false,times.t.<=t)[1]==nothing
#         index=length(times.t)
#         else index=indexin(false,times.t.<=t)[1]-1
#     end
#     t=times.t[index]
#     p1 = plot(title="temps t=$t, ϵ=$epsilon")
#
#     for modele in range(1,size(Us)[end])
#
#         (hhat,uhat)=(Us[modele][:,1,index],Us[modele][:,2,index])
#         (h,u)=(real(ifft((Gamma.^s).*hhat)),real(ifft(uhat)))
#
#
#         p1 = plot!(mesh.x,h)
#     end
#
#     p2 = plot()
#
#     for modele in range(1,size(Us)[end])
#
#         (hhat,uhat)=(Us[modele][:,1,index],Us[modele][:,2,index])
#         (h,u)=(real(ifft((Gamma.^s).*hhat)),real(ifft(uhat)))
#
#         p2 = plot!(fftshift(mesh.k),log10.(1e-18.+abs.(fftshift(hhat))))
#
#     end
#     p=plot(p1,p2,layout=(2,1),label=Labels)
#
#     p
# end
#
#
# function fig(t, times, Gamma, Modeles::Dict, epsilon, mesh)
#
#     Labels = keys(Modeles)
#     s = 0
#     if indexin(false,times.t.<=t)[1]==nothing
#         index=length(times.t)
#     else
#         index=indexin(false,times.t.<=t)[1]-1
#     end
#     t=times.t[index]
#
#     p = plot(layout=(2,1))
#
#     for label in Labels
#         (hhat,uhat)=Modeles[label][index]
#         (h,u)=(real(ifft((Gamma.^s).*hhat)),real(ifft(uhat)))
#         plot!(p[1,1], mesh.x,h; label=string(label))
#         plot!(p[2,1], fftshift(mesh.k),log10.(1e-18.+abs.(fftshift(hhat))); label=string(label))
#     end
#
#     p
# end
#
#
# function fig(t, times::Times, models, mesh::Mesh)
#
#     s = 0
#     index = length(times.t)
#     t = times.t[index]
#
#     p = plot(layout=(2,1), title = "Shallow Water Models")
#
#     hr = zeros(Float64, mesh.N)
#     ur = zeros(Float64, mesh.N)
#
#     for model in models
#         (hhat,uhat) = model.data[index]
#         hr .= real(ifft((model.Gamma.^s).*hhat))
#         ur .= real(ifft(uhat))
#         plot!(p[1,1], mesh.x,hr; label=model.label)
#         plot!(p[2,1], fftshift(model.mesh.k),log10.(1e-18.+abs.(fftshift(hhat))); label=model.label)
#     end
#
#     p
#
# end
#
# function plot_model(times::Times, model::AbstractModel, mesh::Mesh)
#
#     s = 0
#     index = length(times.t)
#     t = times.t[index]
#
#     p = plot(layout=(2,1))
#
#     (hhat,uhat) = model.data[index]
#     (hr,ur)     = (real(ifft((model.Gamma.^s).*hhat)),real(ifft(uhat)))
#     plot!(p[1,1], mesh.x,hr; label=model.label)
#     plot!(p[2,1], fftshift(model.mesh.k),log10.(1e-18.+abs.(fftshift(hhat))); label=model.label)
#
#     p
#
# end
#
# function plot_model!(p, times::Times, model::AbstractModel, mesh::Mesh)
#
#     s = 0
#     index = length(times.t)
#     t = times.t[index]
#
#     (hhat,uhat) = model.data[index]
#     (hr,ur)     = (real(ifft((model.Gamma.^s).*hhat)),real(ifft(uhat)))
#     plot!(p[1,1], mesh.x,hr; label=model.label)
#     plot!(p[2,1], fftshift(model.mesh.k),log10.(1e-18.+abs.(fftshift(hhat))); label=model.label)
#
# end
