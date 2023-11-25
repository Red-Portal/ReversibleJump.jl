
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

struct SinusoidFixedOrderModel{Model <: SinusoidModel}
    k::Int
    model::Model
end

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::SinusoidModel)
    @unpack y, gamma0, nu0, delta, orderprior = model

    N  = length(y)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = delta*delta

    D   = SinusoidDetection.spectrum_matrix(ω, N)
    DᵀD = PDMats.PDMat(Hermitian(D'*D) + 1e-15*I)
    y   = rand(rng, MvNormal(Zeros(N), σ²*(δ²*PDMats.X_invA_Xt(DᵀD, D) + I)))
    ω, y
end

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::SinusoidFixedOrderModel)
    @unpack y, gamma0, nu0, delta, orderprior = model.model

    N  = length(y)
    k  = model.k
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = delta*delta

    D   = SinusoidDetection.spectrum_matrix(ω, N)
    DᵀD = PDMats.PDMat(Hermitian(D'*D) + 1e-15*I)
    y   = rand(rng, MvNormal(Zeros(N), σ²*(δ²*PDMats.X_invA_Xt(DᵀD, D) + I)))
    ω, y
end

include("imhrwmh.jl")
include("rjmcmc.jl")
