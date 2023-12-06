
using Accessors
using AbstractMCMC
using Distributions
using FillArrays
using MCMCTesting
using Random
using SimpleUnPack
#using StatsBase
using LogExpFunctions
using Test

using ReversibleJump

# Models used in the tests
include("models/discrete.jl")
#include("models/utils.jl")

# Tests
include("utils.jl")
include("ais.jl")
include("indep.jl")
include("jump.jl")
include("rjmcmc.jl")
include("nrjmcmc.jl")
include("sample.jl")
include("modelposterior.jl")
