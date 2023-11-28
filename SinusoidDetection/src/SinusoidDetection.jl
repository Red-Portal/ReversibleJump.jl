
module SinusoidDetection

export
    SinusoidKnownSNR,
    rand_sinusoids,
    SinusoidUniformLocalProposal,
    IMHRWMHSinusoidKnownSNR,
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

abstract type AbstractSinusoidModel <: AbstractMCMC.AbstractModel end

function rand_sinusoids(
    rng::Random.AbstractRNG, N::Int, gamma0::Real, nu0::Real, delta::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = delta*delta

    D   = spectrum_matrix(ω, N)
    DᵀD = PDMats.PDMat(Hermitian(D'*D) + 1e-15*I)
    y   = rand(rng, MvNormal(Zeros(N), σ²*(δ²*PDMats.X_invA_Xt(DᵀD, D) + I)))
    SinusoidKnownSNR(y, gamma0, nu0, delta, orderprior)
end

function rand_sinusoids(
    N::Int, gamma0::Real, nu0::Real, delta::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    rand_sinusoids(Random.default_rng(), N, gamma0, nu0, delta, orderprior)
end

include("inference/imhrwmh.jl")
include("inference/slice.jl")

include("models/common.jl")
include("models/knownsnr.jl")
#include("models/unknownsnr.jl")

end
