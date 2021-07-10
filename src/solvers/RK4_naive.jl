export RK4_naive
export step!

"""
    RK4_naive()

Runge-Kutta fourth order solver.

A naive version of `RK4`, without argument since no pre-allocation is performed.

"""
struct RK4_naive <: TimeSolver end

function step!(s  :: RK4_naive,
               m :: AbstractModel ,
               U  ,
               dt )


    U0 = copy(U)
    m.f!( U0 )
    U1 = copy(U0)

    U0 .= U .+ dt/2 .* U1
    m.f!( U0 )
    U2 = copy(U0)

    U0 .= U .+ dt/2 .* U2
    m.f!( U0 )
    U3 = copy(U0)

    U0 .= U .+ dt .* U3
    m.f!( U0 )
    U4 = copy(U0)

    U .+= dt/6 .* (U1 + 2*U2 + 2*U3 + U4 )

end
