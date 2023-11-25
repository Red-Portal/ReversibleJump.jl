
module SinusoidDetection

export
    SinusoidModel,
    rand_sinusoids,
    SinusoidUniformLocalProposal,
    IMHRWMHSinusoid,
    SliceDoublingOut,
    SliceSteppingOut,
    Slice

using AbstractMCMC
using Accessors
using Distributions
using FillArrays
using LinearAlgebra
using PDMats
using Random
using ReversibleJump
using SimpleUnPack

struct GibbsObjective{Model, Idx <: Integer, Vec <: AbstractVector}
    model::Model
    idx  ::Idx
    θ    ::Vec
end
 
include("imhrwmh.jl")
include("slice.jl")
include("model.jl")

end
