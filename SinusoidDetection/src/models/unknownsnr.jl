
struct SinusoidUnknownSNR{
    Y <: AbstractVector, F <: Real, P
} <: SinusoidDetection.AbstractSinusoidModel
    y           ::Y
    nu0         ::F
    gamma0      ::F
    alpha_delta2::F
    beta_delta2 ::F
    orderprior  ::P
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

struct IMHRWMHSinusoidUnknownSNR <: AbstractMCMC.AbstractSampler
    n_snapshots::Int
end

function ReversibleJump.transition_mcmc(
    rng::Random.AbstractRNG, mcmc::IMHRWMHSinusoidUnknownSNR, model, θ
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

        δ²    = θ[1]
        ω     = θ[2:end]
        ω′, lp = ReversibleJump.transition_mcmc(
            rng,
            IMHRWMHSinusoidKnownSNR(mcmc.n_snapshots),
            SinusoidKnownSNR(y, nu0, gamma0, δ², orderprior),
            ω   
        )
        δ²′ = sample_gibbs_snr(rng, y, ω′, nu0, gamma0, alpha_delta2, beta_delta2, δ²)
        θ[1]     = δ²′
        θ[2:end] = ω′
        θ, lp
    end
end

