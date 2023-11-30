
module SinusoidDetection

export
    SinusoidKnownSNR,
    SinusoidUnknownSNR,
    SinusoidUnknownSNRReparam,
    rand_sinusoids_knownsnr,
    rand_sinusoids_unknownsnr,
    rand_sinusoids_unknownsnr_reparam,
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

# General inference algorithsm
include("inference/imhrwmh.jl")
include("inference/slice.jl")

# Sinusoid Models
include("models/common.jl")
include("models/knownsnr.jl")
include("models/unknownsnr.jl")
include("models/unknownsnr_reparam.jl")

# Utilities
include("rand_sinusoids.jl")

end
