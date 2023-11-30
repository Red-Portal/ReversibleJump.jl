
struct SinusoidUnknownSNRReparam{
    Y <: AbstractVector, F <: Real, O
} <: SinusoidDetection.AbstractSinusoidModel
    y           ::Y
    nu0         ::F
    gamma0      ::F
    alpha_delta2::F
    beta_delta2 ::F
    orderprior  ::O
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
    ::SinusoidUnknownSNRReparam,
    ::SinusoidUniformLocalProposal,
    θ, j
)
    logpdf(Uniform(0, π), θ[j+1])
end

function ReversibleJump.local_insert(::SinusoidUnknownSNRReparam, θ, j, θj)
    insert!(copy(θ), j+1, θj)
end

function ReversibleJump.local_deleteat(::SinusoidUnknownSNRReparam, θ, j)
    deleteat!(copy(θ), j+1), θ[j+1]
end

