
struct SinusoidKnownSNR{
    Y <: AbstractVector, F <: Real, P
} <: SinusoidDetection.AbstractSinusoidModel
    y         ::Y
    gamma0    ::F
    nu0       ::F
    delta     ::F
    orderprior::P
end

function ReversibleJump.logdensity(model::SinusoidKnownSNR, ω)
    @unpack y, gamma0, nu0, delta, orderprior = model
    k    = length(ω)
    ℓp_y = collapsed_likelihood(y, ω, delta, gamma0, nu0)
    ℓp_k = logpdf(orderprior, k)
    ℓp_θ = k*logpdf(Uniform(0, π), π/2)
    ℓp_y + ℓp_k + ℓp_θ
end

function ReversibleJump.transition_mcmc(
    rng::Random.AbstractRNG, mcmc::IMHRWMHSinusoid, model, θ)
    σ    = 1/5/mcmc.n_snapshots
    q    = Uniform(0, π)
    θ    = copy(θ)
    k    = length(θ)
    idxs = randperm(rng, k) # the kernel is not reversible without random permutation
    for idx in idxs
        model_gibbs = GibbsObjective(model, idx, θ)
        θ′idx, _ = if rand(Bernoulli(0.2))
            transition_imh(rng, model_gibbs, q, θ[idx])
        else
            transition_rwmh(rng, model_gibbs, σ, θ[idx])
        end
        θ[idx]  = θ′idx
    end
    θ, logdensity(model, θ)
end

