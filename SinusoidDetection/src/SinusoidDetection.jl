
module SinusoidDetection

export
    SinusoidModel,
    rand_sinusoids,
    SinusoidUniformLocalProposal,
    IMHRWMHSinusoid

using AbstractMCMC
using Accessors
using Distributions
using FillArrays
using LinearAlgebra
using PDMats
using Random
using ReversibleJump
using SimpleUnPack
 
include("imhrwmh.jl")
include("slice.jl")
include("model.jl")

end
