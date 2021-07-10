using ProgressMeter
export Problem

"""
    Problem( model, initial, param ; verbose=true)
or
    Problem( model, initial, param, solver ; verbose=true)

Builds an initial-value problem which can then be solved (integrated in time) through `solve!( problem )`

# Arguments
- `model   :: AbstractModel`,  the system of equation solved.
May be built, e.g., by `WaterWaves(param)`;
- `initial :: InitialData`, the initial data.
May be buit, e.g., by `Init(η,v)` where
`η` is the surface deformation and `v` the derivative of the trace of the velocity potential at the surface;
- `param   :: NamedTuple`, must contain values for
    - `N`, the number of collocation points of the spatial grid
    - `L`, the half-length of the spatial grid
    - `T`, the final time of integration
    - `dt`, the timestep
    - additionally, it may contain `Ns` the number of computed data or `ns` for storing data every `ns` computation steps (by default, every computed data is stored).

- `solver  :: TimeSolver`, the solver for time integration (optional, default is explicit Runge-Kutta fourth order solver).
May be built, e.g., by `RK4(model)` or `RK4_naive()`.

Information are not printed if keyword `verbose = false` (default is `true`).

"""
mutable struct Problem

    model   :: AbstractModel
    initial :: InitialData
    param   :: NamedTuple
    solver  :: TimeSolver
    times   :: Times
    mesh    :: Mesh
    data    :: Data

    function Problem(model   :: AbstractModel,
                     initial :: InitialData,
                     param   :: NamedTuple;
                     solver = RK4(model)  :: TimeSolver,
                     verbose = true :: Bool)

        if verbose == true
            @info string("\nBuilds the initial-value problem for model ",model.label,"\n",
                         "with parameters\n",param)
        end
        if in(:Ns,keys(param))
            times = Times(param.dt, param.T; Ns = param.Ns)
        elseif in(:ns,keys(param))
            times = Times(param.dt, param.T; ns = param.ns)
        else
            times = Times(param.dt, param.T)
        end

        mesh  = Mesh(param)

        data  = Data(model.mapto(initial))

        # A basic check
        try
            step!(solver, model, copy(last(data.U)), 1)
        catch
            @warn "The model and the solver are incompatible. solve! will not work."
        end

        new(model, initial, param, solver, times, mesh, data)

    end

end

export solve!

"""
    solve!( problem; verbose=true )

Solves (i.e. integrates in time) an initial-value problem

The argument `problem` should be of type `Problem`.
It may be buit, e.g., by `Problem(model, initial, param)`

Information are not printed if keyword `verbose = false` (default is `true`).

"""
function solve!(problem :: Problem;verbose=true::Bool)

    if verbose == true
        @info string("\nNow solving the initial-value problem for model ",problem.model.label,"\n",
            "with parameters\n",problem.param)
    end

    U = copy(last(problem.data.U))

    dt     = problem.times.dt
    solver = problem.solver
    model  = problem.model
    data   = problem.data.U

    if problem.times.tc == problem.times.ts
        @showprogress 1 for j in 1:problem.times.Ns-1
            step!(solver, model, U, dt)
            push!(data,copy(U))
        end

    elseif length(problem.times.ts) > 25
        @showprogress 1 for j in 1:problem.times.Ns-1
            for l in 1:problem.times.ns[j]
                step!(solver, model, U, dt)
            end
            push!(data,copy(U))
        end
    else
        for j in 1:problem.times.Ns-1
            @showprogress string("Step ",j,"/",problem.times.Ns-1,"...") 1 for l in 1:problem.times.ns[j]
                step!(solver, model, U, dt)
            end
            push!(data,copy(U))
            println()

        end
    end

    println()

end



using Base.Threads

"""
    solve!( problems; verbose=true )

Solves (i.e. integrates in time) a collection of initial-value problems.

The argument `problems` should be a collection (list, array...) of elements of type `Problem`.

"""
function solve!(problems; verbose=true::Bool)
    U=[];nsteps=0
    for problem in problems
        push!(U,copy(last(problem.data.U)))
        nsteps+=problem.times.Ns-1
    end
    pg=Progress(nsteps;dt=1)
    @threads for i in 1:length(problems)
        #pg[i]=Progress(length(Jn[i]);dt=1)
        if problems[i].times.ns == 1

            for j in 1:problems[i].times.Ns-1
                step!(problems[i].solver, problems[i].model, U[i], problems[i].times.dt)
                push!(problems[i].data.U,copy(U[i]))
                next!(pg)
            end

        else
            for j in 1:problems[i].times.Ns-1
                for l in 1:problems[i].times.ns[j]
                    step!(problems[i].solver, problems[i].model, U[i], problems[i].times.dt)
                end
                push!(problems[i].data.U,copy(U[i]))
                next!(pg)
            end
        end
        if verbose == true
            @info string("\nDone solving the model ",problems[i].model.label,"\n",
                "with parameters\n",problems[i].param)
        end

    end

    println()

end
