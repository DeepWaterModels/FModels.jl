export Problem

"""
    Problem( model, initial, param, solver)

- model   : CGBSW or Matsuno
- initial : BellCurve
- param   : must contain N, L, T, dt for Mesh and Times, may contain additional data for Models (ϵ)
- solver  : RK4 (optional)

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
                     param   :: NamedTuple,
                     solver  :: TimeSolver)

        if in(:nr,keys(param))
            times = Times(param.dt, param.T, param.nr)
        else
            times = Times(param.dt, param.T)
        end

        mesh  = Mesh(param)

        data  = Data(mapto(model, initial))

        new(model, initial, param, solver, times, mesh, data)

    end

    function Problem( model   :: AbstractModel,
                      initial :: InitialData,
                      param   :: NamedTuple)

        if in(:nr,keys(param))
            times = Times(param.dt, param.T, param.nr)
        else
            times = Times(param.dt, param.T)
        end
        mesh   = Mesh(param)
        data   = Data(mapto(model,initial))
        solver = RK4(param,model)

        new(model,initial,param,solver,times,mesh,data)

    end

end

