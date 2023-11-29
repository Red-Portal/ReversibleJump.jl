
module SinusoidDetection

export
    SinusoidKnownSNR,
    SinusoidUnknownSNR,
    rand_sinusoids_knownsnr,
    rand_sinusoids_unknownsnr,
    SinusoidUniformLocalProposal,
    IMHRWMHKnownSNR,
    IMHRWMHUnknownSNR,
    SliceDoublingOut,
    SliceSteppingOut,
    Slice,
    SliceKnownSNR,
    SliceUnknownSNR

using AbstractMCMC
using Accessors, SimpleUnPack
using Random, Distributions
using FillArrays
using LinearAlgebra
using PDMats
using ReversibleJump
using LoopVectorization
using FFTW

struct GibbsObjective{Model, Idx <: Integer, Vec <: AbstractVector}
    model::Model
    idx  ::Idx
    θ    ::Vec
end

abstract type AbstractSinusoidModel <: AbstractMCMC.AbstractModel end

include("inference/imhrwmh.jl")
include("inference/slice.jl")

include("models/common.jl")
include("models/knownsnr.jl")
include("models/unknownsnr.jl")

include("rand_sinusoids.jl")

end
