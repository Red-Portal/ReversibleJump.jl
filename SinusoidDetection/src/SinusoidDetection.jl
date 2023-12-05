
module SinusoidDetection

export
    SinusoidKnownSNR,
    SinusoidUnknownSNR,
    SinusoidUnknownSNRReparam,
    rand_sinusoids_knownsnr,
    rand_sinusoids_unknownsnr,
    rand_sinusoids_unknownsnr_reparam,
    SinusoidLocalProposal,
    IMHRWMHSinusoid,
    SliceDoublingOut,
    SliceSteppingOut,
    Slice,
    SliceSinusoid

using AbstractMCMC
using Accessors, SimpleUnPack
using FFTW
using FillArrays
using LinearAlgebra
using LoopVectorization
using PDMats
using ProgressMeter
using Random, Distributions
using ReversibleJump

# Used only by relabel.jl
using Statistics
using StatsFuns

struct GibbsObjective{Model, Idx <: Integer, Vec <: AbstractVector}
    model::Model
    idx  ::Idx
    θ    ::Vec
end

abstract type AbstractSinusoidModel <: AbstractMCMC.AbstractModel end

# General inference algorithsm

include("inference/imhrwmh.jl")
include("inference/slice.jl")

struct IMHRWMHSinusoid{M <: AbstractSinusoidModel} <: AbstractMCMC.AbstractSampler
    model::M
end

struct SliceSinusoid{
    S <: AbstractSliceSampling,
    M <: AbstractSinusoidModel
} <: AbstractMCMC.AbstractSampler
    sampler::S
    model::M
end

Slice()            = Slice(Float64[])
SliceSteppingOut() = SliceSteppingOut(Float64[])
SliceDoublingOut() = SliceDoublingOut(Float64[])

# Sinusoid Models
include("models/common.jl")
include("models/knownsnr.jl")
include("models/unknownsnr.jl")
include("models/unknownsnr_reparam.jl")

# Utilities
include("rand_sinusoids.jl")
include("relabel.jl")

end
