
struct SinusoidKnownSNR{
    Y <: AbstractVector, R <: Real, P, F
} <: SinusoidDetection.AbstractSinusoidModel
    y         ::Y
    nu0       ::R
    gamma0    ::R
    delta2    ::R
    orderprior::P
    freqprop  ::F
end

function SinusoidKnownSNR(
    y          ::AbstractVector,
    ν0         ::Real,
    γ0         ::Real,
    δ²         ::Real,
    orderprior,
)
    ν0, γ0, δ² = promote(ν0, γ0, δ²)
    energydist = spectrum_energy_proposal(y)
    SinusoidKnownSNR(
        y, ν0, γ0, δ², orderprior, energydist
    )
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
    model::SinusoidKnownSNR,
         ::SinusoidLocalProposal,
    θ, j
)
    logpdf(model.freqprop, θ[j])
end

function ReversibleJump.local_insert(::SinusoidKnownSNR, θ, j, θj)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(::SinusoidKnownSNR, θ, j)
    deleteat!(copy(θ), j), θ[j]
end

function ReversibleJump.transition_mcmc(
    rng::Random.AbstractRNG, mcmc::IMHRWMHSinusoid{<:SinusoidKnownSNR}, target, θ
)
    model = mcmc.model
    σ_rw  = 1/5/length(model.y)
    q_imh = model.freqprop

    θ = copy(θ)
    k = length(θ)
    for idx in 1:k
        model_gibbs = GibbsObjective(target, idx, θ)
        θ′idx, _ = if rand(Bernoulli(0.2))
            transition_imh(rng, model_gibbs, q_imh, θ[idx])
        else
            transition_rwmh(rng, model_gibbs, σ_rw, θ[idx])
        end
        θ[idx]  = θ′idx
    end
    θ, logdensity(target, θ)
end

function SliceSinusoid(
    sampler    ::AbstractSliceSampling,
    model      ::SinusoidKnownSNR,
    freq_window::Real
)
    adapted_sampler = @set sampler.window = [freq_window]
    SliceSinusoid(adapted_sampler, model)
end

function ReversibleJump.transition_mcmc(
    rng  ::Random.AbstractRNG,
    mcmc ::SliceSinusoid{<:AbstractSliceSampling, <:SinusoidKnownSNR},
    model,
    θ
)
    sampler         = mcmc.sampler
    sampler_adapted = @set sampler.window = fill(only(sampler.window), length(θ))
    slice_sampling(rng, sampler_adapted, model, θ)
end

