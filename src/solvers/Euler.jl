export Euler
export step!

"""
    Euler(arguments;realdata)

Explicit Euler solver.

Constructs an object of type `TimeSolver` to be used in `Problem(model, initial, param; solver::TimeSolver)`

Arguments can be either
0. an object of type `AbstractModel`;
1. an `Array` of size `(N,m)` where `N` is the number of collocation points and `m` the number of data (equations solved);
2. a `Tuple` `(N,m)` as above;
3. an integer `N` and an integer `m` as above (the latter is optional, by default `m=2`).
4. a `NamedTuple` containing a key `N` and an integer `m` (the latter is optional, by default `m=2`).

The keyword argument `realdata` is optional, and determines whether pre-allocated vectors are real- or complex-valued.
By default, they are either determined by the model or the type of the array in case `0.` and `1.`, complex-valued otherwise.

"""
struct Euler <: TimeSolver

    U1 :: Array
    U2 :: Array

    function Euler( U :: Array; realdata=nothing )
        U1 = copy(U)
        U2 = copy(U)
        if realdata==true
            U1 = real.(U1);U2 = real.(U2)
        end
        if realdata==false
            U1 = complex.(U1);U2 = complex.(U2)
        end
        new( U1, U2)
    end

    function Euler( model :: AbstractModel; realdata=nothing )
        U=model.mapto(Init(x->0*x,x->0*x))
        Euler(U; realdata=realdata)
    end
    function Euler( datasize; realdata=false )
        U = zeros(Float64, datasize)
        Euler(U; realdata=realdata)
    end

    function Euler( N::Int, m=2::Int; realdata=false )
        datasize = (N,m)
        Euler(datasize; realdata=realdata)
    end
    function Euler( param::NamedTuple, datasize=2::Int; realdata=false )
        datasize = (param.N,datasize)
        Euler(datasize; realdata=realdata)
    end
end

function step!(solver :: Euler,
                model :: AbstractModel,
                U  ,
                dt )


    solver.U1 .= U
    model.f!( solver.U1 )
    U .+= dt .* solver.U1

end
