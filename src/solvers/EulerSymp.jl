export EulerSymp
export step!

"""
    EulerSymp(arguments;Niter,implicit,realdata)

Symplectic Euler solver.

Constructs an object of type `TimeSolver` to be used in `Problem(model, initial, param; solver::TimeSolver)`

Arguments can be either
0. an object of type `AbstractModel`;
1. an `Array` of size `(N,m)` where `N` is the number of collocation points, and `m` is irrelevant;
2. an integer `N` being the number of collocation points;
3. a `NamedTuple` containing a key `N`.

The keyword argument `Niter` (optional, defaut value = 10) determines the number of steps in the Neumann iteration solver of the implicit step.
The keyword argument `implicit` (optional, defaut value = 1) determines which equation is implicit (must be `1` or `2`).
The keyword argument `realdata` is optional, and determines whether pre-allocated vectors are real- or complex-valued.
By default, they are either determined by the model or the type of the array in case `0.` and `1.`, complex-valued otherwise.


"""
struct EulerSymp <: TimeSolver

    U1 :: Array
    U2   :: Array
    Niter:: Int
    implicit:: Int

    function EulerSymp( U :: Array; Niter = 10, implicit = 1, realdata=nothing )
        U1 = copy(U[:,1])
        U2 = copy(U[:,1])
        if realdata==true
            U1 = real.(U1);U2 = real.(U2)
        end
        if realdata==false
            U1 = complex.(U1);U2 = complex.(U2)
        end
        if implicit!=1 && implicit!=2
            @warn "the keyword `implicit` must be 1 or 2. solve! will not work."
        end
        new( U1, U2, Niter, implicit)
    end

    function EulerSymp( model :: AbstractModel; Niter = 10, implicit = 1, realdata=nothing )
        U=model.mapto(Init(x->0*x,x->0*x))
        EulerSymp(U; Niter = Niter, realdata=realdata, implicit=implicit)
    end
    function EulerSymp( N::Int; Niter=10, implicit = 1, realdata=false )
        U = zeros(Float64, (N,2))
        EulerSymp(U; Niter = Niter, realdata=realdata,implicit=implicit)
    end
    function EulerSymp( param::NamedTuple, datasize=2::Int; Niter = 10, implicit = 1, realdata=false )
        EulerSymp(param.N; Niter = Niter, realdata=realdata,implicit=implicit)
    end
end

function step!(solver :: EulerSymp,
                model :: AbstractModel,
                U  ,
                dt )


    solver.U1 .= copy(U[:,1])
    solver.U2 .= copy(U[:,2])

    if solver.implicit == 1
        for i=1:solver.Niter
            model.f1!( solver.U1, solver.U2 )
            solver.U1 .= U[:,1] + dt * solver.U1
        end
        U[:,1] .= solver.U1
        model.f2!( solver.U1, solver.U2 )
        U[:,2] .+= dt * solver.U2
    elseif solver.implicit == 2
        for i=1:solver.Niter
            model.f2!( solver.U1, solver.U2 )
            solver.U2 .= U[:,2] + dt * solver.U2
        end
        U[:,2] .= solver.U2
        model.f1!( solver.U1, solver.U2 )
        U[:,1] .+= dt * solver.U1
    else
        error("when defining `EulerSymp`, the keyword `implicit` must be either 1 or 2.")
    end
end
