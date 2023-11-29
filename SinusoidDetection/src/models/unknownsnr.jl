
struct SinusoidUnknownSNR{
    Y <: AbstractVector, F <: Real, O
} <: SinusoidDetection.AbstractSinusoidModel
    y           ::Y
    nu0         ::F
    gamma0      ::F
    alpha_delta2::F
    beta_delta2 ::F
    orderprior  ::O
end

function ReversibleJump.logdensity(model::SinusoidUnknownSNR, θ)
    @unpack y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior = model
    δ² = θ[1]
    ω  = θ[2:end]
    k  = length(ω)

    if δ² < eps(eltype(θ))
        -Inf
    else
        ℓp_y  = collapsed_likelihood(y, ω, δ², nu0, gamma0)
        ℓp_δ² = logpdf(InverseGamma(alpha_delta2, beta_delta2), δ²)
        ℓp_k  = logpdf(orderprior, k)
        ℓp_θ  = k*logpdf(Uniform(0, π), π/2)
        ℓp_y + ℓp_k + ℓp_θ + ℓp_δ²
    end
end

function ReversibleJump.local_proposal_logpdf(
    ::SinusoidUnknownSNR,
    ::SinusoidUniformLocalProposal,
    θ, j
)
    logpdf(Uniform(0, π), θ[j+1])
end

function ReversibleJump.local_insert(::SinusoidUnknownSNR, θ, j, θj)
    insert!(copy(θ), j+1, θj)
end

function ReversibleJump.local_deleteat(::SinusoidUnknownSNR, θ, j)
    deleteat!(copy(θ), j+1), θ[j+1]
end

struct IMHRWMHUnknownSNR{
    Prop <: ContinuousUnivariateDistribution
} <: AbstractMCMC.AbstractSampler
    indep_proposal::Prop
    n_snapshots   ::Int
end

function IMHRWMHUnknownSNR(y::AbstractVector, n_snapshots::Int)
    q_imh = spectrum_energy_proposal(y, n_snapshots)
    IMHRWMHUnknownSNR(q_imh, n_snapshots)
end

function ReversibleJump.transition_mcmc(
    rng::Random.AbstractRNG, mcmc::IMHRWMHUnknownSNR, model, θ
)
    @unpack y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior = model
    θ = copy(θ)
    k = length(θ) - 1

    if k == 0
        θ[1] = rand(rng, InverseGamma(k + alpha_delta2, beta_delta2))
        θ, logdensity(model, θ)
    else
        #=
            Partially-collapsed Gibbs-sampler:
            ω  ~ p(ω |y, δ²)
            σ² ~ p(σ²|y, δ², ω)     (discarded)
            a  ~ p(a |y, δ², ω, σ²) (discarded)
            δ² ~ p(δ²|y, ω, a, σ²)
         
            The order matters: ω should be sampled first and then δ²
            (For as why, refer to Sampler 3 in Van Dyk and Park (2008), JASA.)
        =##
        q_imh = mcmc.indep_proposal
        σ_rw  = 1/5/mcmc.n_snapshots
        ω_idx_range = 2:length(θ)
        for ω_idx in ω_idx_range
            model_gibbs = GibbsObjective(model, ω_idx, θ)
            ωi′, _ = if rand(Bernoulli(0.2))
                transition_imh(rng, model_gibbs, q_imh, θ[ω_idx])
            else
                transition_rwmh(rng, model_gibbs, σ_rw, θ[ω_idx])
            end
            θ[ω_idx] = ωi′
        end
        θ[1] = sample_gibbs_snr(
            rng, y, θ[2:end], nu0, gamma0, alpha_delta2, beta_delta2, θ[1]
        )
        θ, logdensity(model, θ)
    end
end

struct SliceUnknownSNR{
    S <: AbstractSliceSampling, F <: Real
} <: AbstractMCMC.AbstractSampler
    slice_sampler::S
    window_omega ::F
    window_delta2::F
end

function ReversibleJump.transition_mcmc(
    rng  ::Random.AbstractRNG,
    mcmc ::SliceUnknownSNR,
    model,
    θ
)
    k             = length(θ) - 1
    window        = vcat([mcmc.window_delta2], fill(mcmc.window_omega, k))
    slice_sampler = mcmc.slice_sampler
    slice_sampling(rng, (@set slice_sampler.window = window), model, θ)
end
