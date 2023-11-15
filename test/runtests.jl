
using AbstractMCMC
using Distributions
using FillArrays
using MCMCTesting
using Random
using StatsBase
using Test

#using ReversibleJump

# Models used in the tests
include("models/categorical.jl")
include("models/gaussian2d.jl")
include("models/utils.jl")

# Tests
include("ais.jl")
include("indep.jl")
include("jump.jl")
include("rjmcmc.jl")
