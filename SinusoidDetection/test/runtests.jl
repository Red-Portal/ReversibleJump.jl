
using AbstractMCMC
using MCMCTesting

using Accessors
using Distributions
using FillArrays
using LinearAlgebra
using PDMats
using Random
using ReversibleJump
using SimpleUnPack
using SinusoidDetection

using Test

struct SinusoidFixedOrderModel{Model <: SinusoidDetection.AbstractSinusoidModel}
    k::Int
    model::Model
end

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::SinusoidKnownSNR)
    @unpack y, nu0, gamma0, delta2, orderprior = model

    N  = length(y)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², delta2)
    ω, y
end

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::SinusoidUnknownSNR)
    @unpack y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior = model

    N  = length(y)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = rand(rng, InverseGamma(alpha_delta2, beta_delta2))
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², δ²)
    θ  = vcat([δ²], ω)
    θ, y
end

function MCMCTesting.sample_joint(
    rng::Random.AbstractRNG, model::SinusoidFixedOrderModel{<:SinusoidKnownSNR}
)
    @unpack y, gamma0, nu0, delta2, orderprior = model.model

    N  = length(y)
    k  = model.k
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², delta2)
    ω, y
end

function MCMCTesting.sample_joint(
    rng::Random.AbstractRNG, model::SinusoidFixedOrderModel{<:SinusoidUnknownSNR}
)
    @unpack y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior = model.model

    N  = length(y)
    k  = model.k
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = rand(rng, InverseGamma(alpha_delta2, beta_delta2))
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², δ²)
    θ  = vcat([δ²], ω)
    θ, y
end

include("slice.jl")
include("imhrwmh.jl")
include("rjmcmc.jl")
