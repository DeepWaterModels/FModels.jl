export interpolate,solution

"""
    interpolate(mesh,vector;n=2^4)

Interpolate a vector `vector` defined on a uniform collocation grid defined by `mesh`.

Returns `(new_mesh,new_vector)` a new uniform mesh with `n` times as many values, and the vector of values on collocation points.

"""
function interpolate(mesh::Mesh,vector;n=2^3::Int)

    fourier=fft(vector)
    m=Int(mesh.N/2)
    new_fourier = [fourier[1:m] ;zeros((n-1)*2*m) ;fourier[m+1:end]]


    new_mesh = Mesh(mesh.xmin,mesh.xmax,n*mesh.N)
    new_vector=ifft(new_fourier)*n
    if all((vector[i] isa Real) for i in length(vector))
        new_vector=real.(new_vector)
    end
    return new_mesh,new_vector
end

"""
    interpolate(mesh,vector,x)

Interpolate a vector `vector` defined on a uniform collocation grid defined by `mesh`, on collocation points given by `x`.

Returns `new_vector` the vector of values on collocation points.

"""
function interpolate(mesh::Mesh,vector,x)

    fourier=fft(vector)
    k = mesh.k
    x₀ = mesh.xmin

    new_vector=exp.(1im*(x.-x₀)*k')*fourier/length(k)
    if all((vector[i] isa Real) for i in length(vector))
        new_vector=real.(new_vector)
    end
    return new_vector
end


"""
    solution(p::Problem;t,x,interpol)

Gives the solution of a solved initial-value at a given time `t`.

# Arguments
- Argument `p` is of type `Problem`.
- Keyword argument `t` is optional, the last computed time is returned by default.
- Keyword argument `x` is optional, if provided the solution is interpolated to the collocation vector `x`.
- Keyword argument `interpol` is optional, if an integer is provided the solution is interpolated on as many collocation points (if `true`, then the default value `2^3` is chosen).


# Return values
Provides `(η,v,x,t)` where
- `η` is the surface deformation at collocation points;
- `v` is the tangential velocity (derivative of the trace of the velocity potential) at collocation points;
- `x` is the vector of collocation points;
- `t` the time (first computed time greater or equal to provided `t`).

"""

function solution(p::Problem; t=nothing, x=nothing, interpol = false)
	if t == nothing t = p.times.tfin end
	t=min(max(t,0),p.times.tfin)
	index = indexin(false,p.times.ts.<t)[1]
	t=p.times.ts[index]
	if p.model.label=="water waves"
		if x != nothing
			@error "cannot interpolate non-regularly spaced mesh."
		end
		(x,η,v) = (p.model.mapfro)(p.data.U[index])

	else
    	(η,v) = (p.model.mapfro)(p.data.U[index])
		mesh=p.mesh
		if x != nothing
			η = interpolate(mesh,η,x)
			v = interpolate(mesh,v,x)
		else
			x=mesh.x
		end
		if interpol == true
			new_mesh,η = interpolate(mesh,η)
			new_mesh,v = interpolate(mesh,v)
			x = new_mesh.x
		elseif isa(interpol,Int)
			new_mesh,η = interpolate(mesh,η;n=interpol)
			new_mesh,v = interpolate(mesh,v;n=interpol)
			x = new_mesh.x
		end
	end
	return (η,v,x,t)
end
