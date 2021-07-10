export include_all

using ProgressMeter
using FFTW
#using LinearAlgebra
ENV["PLOTS_USE_ATOM_PLOTPANE"] = "false"
# Note that javascript-based libraries (for example: PlotlyJS) cannot be shown in the PlotPane due to issues within Atom's internals.
#using Plots
#gr()
#pyplot()
#plotlyjs()

include("types.jl")
include("solvers/RK4.jl")
include("tools.jl")


function include_all(dir=nothing)
    if dir!=nothing
        dir=string("src/",dir)
        for file in readdir(abspath(dir),join=true)
            try include(file)
                @info string("included ",file)
            catch
                @info string("did not include ",file)
            end
        end
    else
        include_all("solvers")
        include_all("initialdata")
        include_all("models")
        include(abspath("src/Figures.jl"))
        include(abspath("src/LoadSave.jl"))
    end
end

#include_all()

# include("solvers/RK4.jl")
# include("solvers/RK4_naive.jl")
# include("models/CGBSW.jl")
# include("models/CGBSW_naive.jl")
# include("models/Matsuno.jl")
# include("models/Matsuno_naive.jl")
# include("models/Matsuno_mod_naive.jl")
# include("models/Boussinesq.jl")
# include("models/PseudoSpectral.jl")
# include("models/WaterWaves.jl")
# include("models/WhithamBoussinesq.jl")
# include("models/WhithamGreenNaghdi.jl")
# include("models/WhithamGreenNaghdiGPU.jl") #comment this line if you have problems with CuArrays
#
