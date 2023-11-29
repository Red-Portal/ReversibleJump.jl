
struct SinusoidKnownSNR{
    Y <: AbstractVector, F <: Real, P
} <: SinusoidDetection.AbstractSinusoidModel
    y         ::Y
    nu0       ::F
    gamma0    ::F
    delta2    ::F
    orderprior::P
end

function ReversibleJump.logdensity(model::SinusoidKnownSNR, ω)
    @unpack y, gamma0, nu0, delta2, orderprior = model
    k    = length(ω)
    ℓp_y = collapsed_likelihood(y, ω, delta2, nu0, gamma0)
    ℓp_k = logpdf(orderprior, k)
    ℓp_θ = k*logpdf(Uniform(0, π), π/2)
    ℓp_y + ℓp_k + ℓp_θ
end

function ReversibleJump.local_proposal_logpdf(
    ::SinusoidKnownSNR,
    ::SinusoidUniformLocalProposal,
    θ, j
)
    logpdf(Uniform(0, π), θ[j])
end

function ReversibleJump.local_insert(::SinusoidKnownSNR, θ, j, θj)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(::SinusoidKnownSNR, θ, j)
    deleteat!(copy(θ), j), θ[j]
end

struct IMHRWMHKnownSNR{
    Prop <: ContinuousUnivariateDistribution
} <: AbstractMCMC.AbstractSampler
    indep_proposal::Prop
    n_snapshots   ::Int
end

function IMHRWMHKnownSNR(y::AbstractVector, n_snapshots::Int)
    q_imh = spectrum_energy_proposal(y, n_snapshots)
    IMHRWMHKnownSNR(q_imh, n_snapshots)
end

function ReversibleJump.transition_mcmc(
    rng::Random.AbstractRNG, mcmc::IMHRWMHKnownSNR, model, θ
)
    σ_rw  = 1/5/mcmc.n_snapshots
    q_imh = mcmc.indep_proposal

    θ = copy(θ)
    k = length(θ)
    for idx in 1:k
        model_gibbs = GibbsObjective(model, idx, θ)
        θ′idx, _ = if rand(Bernoulli(0.2))
            transition_imh(rng, model_gibbs, q_imh, θ[idx])
        else
            transition_rwmh(rng, model_gibbs, σ_rw, θ[idx])
        end
        θ[idx]  = θ′idx
    end
    θ, logdensity(model, θ)
end

struct SliceKnownSNR{
    S <: AbstractSliceSampling, F <: Real
} <: AbstractMCMC.AbstractSampler
    slice_sampler::S
    window_omega ::F
end

function ReversibleJump.transition_mcmc(
    rng  ::Random.AbstractRNG,
    mcmc ::SliceKnownSNR,
    model,
    θ
)
    window        = mcmc.window_omega
    slice_sampler = mcmc.slice_sampler
    slice_sampling(rng, (@set slice_sampler.window = window), model, θ)
end
