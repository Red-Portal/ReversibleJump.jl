
using AbstractMCMC
using Distributions
using Random
using Test

using ReversibleJump

# Models used in the tests
include("models/categorical.jl")

# Tests
include("proposal_ais.jl")
