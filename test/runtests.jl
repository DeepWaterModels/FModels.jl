ENV["GKSwstype"]="100"

using Test
using ShallowWaterModels
include("../src/initialdata/BellCurve.jl")
#include("../src/models/WhithamGreenNaghdi.jl")
#include("../src/models/PseudoSpectral.jl")
include_all("models") # I do not understand why this line cannot be replaced by the above two
include("../src/LoadSave.jl")
#using JLD

param = ( ϵ  = 1/2, μ = 1, θ = 1)
paramX= ( N  = 2^8, L  = 10)
paramT= ( T  = 5, dt = 0.1)

init     = BellCurve(param)
model    = WhithamGreenNaghdi(merge(param,paramX);ktol=1e-10)
problem1 = Problem( model, init, merge(paramT,paramX); solver = RK4(paramX) )

@testset "LoadSave" begin

    dump  = convert( ProblemSave, problem1 )
    pload = convert( Problem, dump )

    save(problem1, "testsave")
    pload = load("testsave")

    @test true
    @test pload.model.kwargs == problem1.model.kwargs
    @test pload.solver.Uhat == problem1.solver.Uhat

end

@testset "Parameters" begin

    @test param.ϵ  == 0.5
    @test param.μ  == 1
    @test paramX.N  == 256
    @test paramX.L  == 10
    @test paramT.T  == 5
    @test paramT.dt == 0.1
    @test param.θ == 1

end

solve!(problem1)

@testset "Test problem with Whitham-Green-Naghdi model" begin
    @test !any(isnan,problem1.data.U[end][1])
    @test !any(isnan,problem1.data.U[end][2])
end

WW3  = PseudoSpectral(merge(param,paramX);order=3,dealias=1)
problem2 = Problem(WW3, init, merge(paramX,paramT) )

solve!( problem2 )

@testset "Test problem with Pseudo-spectral model" begin
    @test !any(isnan,problem2.data.U[end][1])
    @test !any(isnan,problem2.data.U[end][2])
end
