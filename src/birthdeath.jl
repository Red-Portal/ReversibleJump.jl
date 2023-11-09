
function proposal_ais(
    rng::Random.AbstractRNG, θ, model, ℓπ, G⁻¹, ϕ_ktok′, ϕ_k′tok, T, kernel
)
    #=
        Generic Annealed Importance Sampling Jump Proposal

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible Jump MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.

        If this is birth proposal, k_birth is the index of the "newborn".
    =##
    ∑α  = 0.0
    n_α = 0
    ℓr  = -ℓπ - ϕ_ktok′(θ)
    for t = 1:T-1
        target_tempered = (θ_) -> begin
            ℓρ_T = logdensity(model, θ_)      + ϕ_k′tok(θ_)
            ℓρ_0 = logdensity(model, G⁻¹(θ_)) + ϕ_ktok′(θ_)
            #ℓγ  = log(t / T)
            #logaddexp(ℓρ_0 + log1mexp(ℓγ), ℓρ_T + ℓγ)
            γ = (t/T)
            γ*ℓρ_T + (1 - γ)*ℓρ_0
        end
        ℓρₜ = target_tempered(θ)
        if !isfinite(ℓρₜ)
            break
        end
        θ, ℓρₜ′, stat = kernel(rng, target_tempered, θ)
        ∑α  += stat.acceptance_rate
        n_α += 1
        ℓr  += ℓρₜ - ℓρₜ′
    end
    ℓπ′  = logdensity(model, θ)
    ℓr += ℓπ′ + ϕ_k′tok(θ)
    𝔼α  = ∑α/n_α
    θ, ℓr, ℓπ′, (acceptance_rate=𝔼α,)
end

function proposal_ais_birth(
    rng::Random.AbstractRNG, θ, model, ℓπ::Real, T::Int, kernel
)
    #=
        Annealed Importance Sampling Birth Proposal

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible Jump MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.
    =##

    k  = model_order(model, θ)
    θⱼ = local_proposal_sample(rng, model)
    j  = rand(rng, DiscreteUniform(1, k+1))
    θ′ = local_insert(θ, j, θⱼ)
    k′ = model_order(model, θ′)

    G⁻¹(θ_)    = local_deleteat(θ_, j)
    ϕktok′(θ_) = local_proposal_logpdf(model, θ_, j) + log(1/(k + 1))
    ϕk′tok(θ_) = log(1/k′)
    proposal_ais(rng, θ′, model, ℓπ, G⁻¹, ϕktok′, ϕk′tok, T, kernel)
end

function proposal_ais_death(rng::Random.AbstractRNG, θ, model, ℓπ, T, kernel)
    #=
        Annealed Importance Sampling Death Proposal 

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible Jump MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.
    =##

    k  = model_order(model, θ)
    j  = rand(rng, DiscreteUniform(1, k))
    θⱼ = getlocalindex(θ, j)
    θ′  = local_deleteat(θ, j)
    k′  = k - 1

    G⁻¹(θ_)   = local_insert(θ_, j, θⱼ)
    ϕktok′(θ_) = log(1/k)
    ϕk′tok(θ_) = local_proposal_logpdf(model, θ, j) + log(1/(k′ + 1))
    proposal_ais(rng, θ′, model, ℓπ, G⁻¹, ϕktok′, ϕk′tok, T, kernel)
end

