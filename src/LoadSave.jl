export ProblemSave, save, loadpb
using JLD
struct ProblemSave

    param_problem :: Dict
    model   :: Symbol
    param_model   :: Dict
    solver  :: Symbol
    param_solver   :: Dict
    data    :: Data

    function ProblemSave(param_p, model, param_m, solver, param_s, data)
         new( param_p, model, param_m, solver, param_s, deepcopy(data))
    end

end

import Base.convert

function convert(::Type{ProblemSave}, p :: Problem )

    model   = Symbol(typeof(p.model))
    solver  = Symbol(typeof(p.solver))

    param_problem = p.param
    if :param in fieldnames(typeof(p.model))
        param_model=p.model.param
    else
        param_model = NamedTuple()
    end
    if :kwargs in fieldnames(typeof(p.model))
        kwargs=zip(keys(p.model.kwargs),values(p.model.kwargs))
        param_model=merge(param_model,(kwargs=kwargs,))
    end
    if :param in fieldnames(typeof(p.solver))
        param_solver=p.solver.param
    else
        param_solver = NamedTuple()
    end
    param_problem = Dict(pairs(param_problem))
    param_model   = Dict(pairs(param_model))
    param_solver  = Dict(pairs(param_solver))
    data    = p.data

    ProblemSave(param_problem, model, param_model, solver, param_solver, data)

end

function convert(::Type{Problem}, p :: ProblemSave)

    param_p = (;p.param_problem...)
    param_m = (;p.param_model...)
    param_s = (;p.param_solver...)

    # Reconstructs the model
    model = try # try with set of parameters defined by the model
        getfield(ShallowWaterModels, p.model)(param_m)
    catch m
        if isa(m,AbstractModel) == false # if error, then use set of parameters defined by the problem
            model = getfield(ShallowWaterModels, p.model)(param_p)
        end
    end
    if :kwargs in fieldnames(typeof(model))
        model = getfield(ShallowWaterModels, p.model)(param_m;param_m.kwargs...)
    end

    # Reconstructs the initial data
    mesh=Mesh(param_p)
    U=first(p.data.U)
    initial = Init(mesh,U[:,1],U[:,2])

    # Reconstructs the solver (may not work with other user-defined solvers)
    solver = try
        getfield(ShallowWaterModels, p.solver)(param_s)
    catch s
        if isa(s,TimeSolver) == false
            solver = try
                getfield(ShallowWaterModels, p.solver)(model)
            catch s
                if isa(s,TimeSolver) == false
                    solver = try
                        getfield(ShallowWaterModels, p.solver)()
                    catch s
                        if isa(s,TimeSolver) == false
                            @warn "Cannot set up the solver. Choose RK4."
                            solver = RK4(model)
                        end
                    end
                end
            end
        end
    end

    # Reconstructs the structure of the problem
    pb = Problem(model, initial, param_p; solver = solver)

    # Writes the raw data
    pb.data = p.data

    return pb

end

import JLD.save

"""
    `save(p::Problem,name::String)`

Saves the content of an object `Problem` into the file `name.jld`.
"""

function save(p::Problem,name::String)

    filename = string(name,".jld")
    JLD.save(filename, name, convert(ProblemSave,p))

end

import JLD.load

"""
    `loadpb(name::String)`

Loads the contents of the file `name.jld` as a problem of type `:Problem`.
"""
function loadpb(name::String)

    convert(Problem, JLD.load(string(name,".jld"), name ))

end
