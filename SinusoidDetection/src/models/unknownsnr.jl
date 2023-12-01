
struct SinusoidUnknownSNR{
    Y <: AbstractVector, R <: Real, O, F
} <: SinusoidDetection.AbstractSinusoidModel
    y           ::Y
    nu0         ::R
    gamma0      ::R
    alpha_delta2::R
    beta_delta2 ::R
    orderprior  ::O
    freqprop    ::F
end

function SinusoidUnknownSNR(
    y          ::AbstractVector,
    ν0         ::Real,
    γ0         ::Real,
    α_δ²       ::Real,
    β_δ²       ::Real,
    orderprior,
)
    ν0, γ0, α_δ², β_δ² = promote(ν0, γ0, α_δ², β_δ²)
    energydist = spectrum_energy_proposal(y)
    SinusoidUnknownSNR(
        y, ν0, γ0, α_δ², β_δ², orderprior, energydist
    )
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
    model::SinusoidUnknownSNR,
         ::SinusoidLocalProposal,
    θ, j
)
    logpdf(model.freqprop, θ[j+1])
end

function ReversibleJump.local_insert(::SinusoidUnknownSNR, θ, j, θj)
    insert!(copy(θ), j+1, θj)
end

function ReversibleJump.local_deleteat(::SinusoidUnknownSNR, θ, j)
    deleteat!(copy(θ), j+1), θ[j+1]
end

function ReversibleJump.transition_mcmc(
    rng   ::Random.AbstractRNG,
    mcmc  ::IMHRWMHSinusoid{<:SinusoidUnknownSNR},
    target,
    θ
)
    model = mcmc.model

    @unpack y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior = model

    θ = copy(θ)
    k = length(θ) - 1

    if k == 0
        θ[1] = rand(rng, InverseGamma(k + alpha_delta2, beta_delta2))
        θ, logdensity(target, θ)
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
        q_imh = model.freqprop
        σ_rw  = 1/5/length(model.y)
        ω_idx_range = 2:length(θ)
        for ω_idx in ω_idx_range
            model_gibbs = GibbsObjective(target, ω_idx, θ)
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
        θ, logdensity(target, θ)
    end
end

function SliceSinusoid(
    sampler    ::AbstractSliceSampling,
    model      ::SinusoidUnknownSNR,
    freq_window::Real,
    snr_window ::Real
)
    adapted_sampler = @set sampler.window = [freq_window, snr_window]
    SliceSinusoid(adapted_sampler, model)
end

function ReversibleJump.transition_mcmc(
    rng  ::Random.AbstractRNG,
    mcmc ::SliceSinusoid{<:AbstractSliceSampling, <:SinusoidUnknownSNR},
    model,
    θ
)
    sampler         = mcmc.sampler
    window_base     = sampler.window
    window_adapted  = vcat(first(window_base), fill(last(window_base), length(θ)-1))
    sampler_adapted = @set sampler.window = window_adapted
    slice_sampling(rng, sampler_adapted, model, θ)
end
