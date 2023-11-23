
module SinusoidDetection

export
    SinusoidModel,
    SinusoidUniformLocalProposal,
    IMHSinusoid

using AbstractMCMC
using Accessors
using Distributions
using FillArrays
using LinearAlgebra
using PDMats
using Random
using ReversibleJump
using SimpleUnPack
 
include("mcmc.jl")
include("model.jl")

end
