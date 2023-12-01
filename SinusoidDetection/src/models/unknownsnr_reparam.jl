
struct SinusoidUnknownSNRReparam{
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

function SinusoidUnknownSNRReparam(
    y          ::AbstractVector,
    ν0         ::Real,
    γ0         ::Real,
    α_δ²       ::Real,
    β_δ²       ::Real,
    orderprior,
)
    ν0, γ0, α_δ², β_δ² = promote(ν0, γ0, α_δ², β_δ²)
    energydist = spectrum_energy_proposal(y)
    SinusoidUnknownSNRReparam(
        y, ν0, γ0, α_δ², β_δ², orderprior, energydist
    )
end

function ReversibleJump.logdensity(model::SinusoidUnknownSNRReparam, θ)
    @unpack y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior = model
    ℓδ²  = θ[1]
    δ²   = exp(ℓδ²)
    ℓjac = ℓδ²

    ω = θ[2:end]
    k = length(ω)

    ℓp_y  = collapsed_likelihood(y, ω, δ², nu0, gamma0)
    ℓp_δ² = logpdf(InverseGamma(alpha_delta2, beta_delta2), δ²)
    ℓp_k  = logpdf(orderprior, k)
    ℓp_θ  = k*logpdf(Uniform(0, π), π/2)
    ℓp_y + ℓp_k + ℓp_θ + ℓp_δ² + ℓjac
end

function ReversibleJump.local_proposal_logpdf(
    model::SinusoidUnknownSNRReparam,
         ::SinusoidLocalProposal,
    θ, j
)
    logpdf(model.freqprop, θ[j+1])
end

function ReversibleJump.local_insert(::SinusoidUnknownSNRReparam, θ, j, θj)
    insert!(copy(θ), j+1, θj)
end

function ReversibleJump.local_deleteat(::SinusoidUnknownSNRReparam, θ, j)
    deleteat!(copy(θ), j+1), θ[j+1]
end

function SliceSinusoid(
    sampler    ::AbstractSliceSampling,
    model      ::SinusoidUnknownSNRReparam,
    freq_window::Real,
    snr_window ::Real
)
    adapted_sampler = @set sampler.window = [freq_window, snr_window]
    SliceSinusoid(adapted_sampler, model)
end

function ReversibleJump.transition_mcmc(
    rng  ::Random.AbstractRNG,
    mcmc ::SliceSinusoid{<:AbstractSliceSampling, <:SinusoidUnknownSNRReparam},
    model,
    θ
)
    sampler         = mcmc.sampler
    window_base     = sampler.window
    window_adapted  = vcat(first(window_base), fill(last(window_base), length(θ)-1))
    sampler_adapted = @set sampler.window = window_adapted
    slice_sampling(rng, sampler_adapted, model, θ)
end
