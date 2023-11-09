
abstract type AbstractAnnealingPath end

struct GeometricPath <: AbstractAnnealingPath end

struct ArithmeticPath <: AbstractAnnealingPath end

struct AnnealedJumpProposal{
    Prop,
    AnnealPath <: AbstractAnnealingPath
} <: AbstractJumpProposal
    n_annealed    ::Int
    local_proposal::Prop
    path          ::AnnealPath
end

struct AnnealedTarget{
    Model,
    INV,
    FWD,
    BWD,
    AnnealPath <: AbstractAnnealingPath
}
    model      ::Model
    t          ::Int
    T          ::Int
    inverse_map::INV
    fwd_density::FWD
    bwd_density::BWD
    path       ::AnnealPath
end

function anneal(::GeometricPath, ℓρ_T, ℓρ_0, t, T)
    γ = (t/T)
    γ*ℓρ_T + (1 - γ)*ℓρ_0
end

function anneal(::ArithmeticPath, ℓρ_T, ℓρ_0, t, T)
    ℓγ  = log(t/T)
    logaddexp(ℓρ_0 + log1mexp(ℓγ), ℓρ_T + ℓγ)
end

function logdensity(annealed::AnnealedTarget, θ)
    @unpack model, t, T, inverse_map, fwd_density, bwd_density, path = annealed
    ℓρ_T = logdensity(model, θ) + bwd_density(θ)
    ℓρ_0 = logdensity(model, inverse_map(θ)) + fwd_density(θ)
    anneal(path, ℓρ_T, ℓρ_0, t, T)
end

function propose_ais(
    rng   ::Random.AbstractRNG,
    jump  ::AnnealedJumpProposal,
    mcmc,
    model,
    θ,
    ℓπ,
    G⁻¹,
    ϕ_ktok′,
    ϕ_k′tok,
)
    #=
        Generic Annealed Importance Sampling Jump Proposal

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible Jump MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.
    =##
    @unpack n_annealed, path = jump 

    ℓr  = -ℓπ - ϕ_ktok′(θ)
    T   = n_annealed 

    target = AnnealedTarget(model, 0, T, G⁻¹, ϕ_ktok′, ϕ_k′tok, path)
    for t = 1:T-1
        target_annealed = @set target.t = t

        ℓρₜ = logdensity(target_annealed, θ)
        if !isfinite(ℓρₜ)
            break
        end
        θ, ℓρₜ′ = mcmc_step(rng, mcmc, target_annealed, θ)
        ℓr    += ℓρₜ - ℓρₜ′
    end
    ℓπ′  = logdensity(model, θ)
    ℓr += ℓπ′ + ϕ_k′tok(θ)
    θ, ℓπ′, ℓr, NamedTuple()
end

function propose_jump(
    rng     ::Random.AbstractRNG,
            ::Birth,
    proposal::AnnealedJumpProposal,
    prev    ::RJState,
    mcmc,
    model,
)
    #=
        Annealed Importance Sampling Birth Proposal

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible Jump MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.
    =##
    ℓπ = prev.lp
    θ  = prev.param
    k  = prev.order

    k′ = k + 1
    j  = rand(rng, DiscreteUniform(1, k + 1))

    newborn = local_proposal_sample(rng, model, proposal.local_proposal)
    θ′       = local_insert(model, θ, j, newborn)

    G⁻¹(θ_)    = local_deleteat(model, θ_, j)
    ϕktok′(θ_) = local_proposal_logpdf(model, proposal.local_proposal, θ_, j) + log(1/(k + 1))
    ϕk′tok(θ_) = log(1/k′)

    θ, ℓπ′, ℓr, stat = propose_ais(rng, proposal, mcmc, model, θ′, ℓπ, G⁻¹, ϕktok′, ϕk′tok)
    RJState(θ, ℓπ′, k′, stat), ℓr
end

function propose_jump(
    rng     ::Random.AbstractRNG,
            ::Death,
    proposal::AnnealedJumpProposal,
    prev    ::RJState,
    mcmc,
    model,
)
    #=
        Annealed Importance Sampling Death Proposal 

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible proposal MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.
    =##
    ℓπ = prev.lp
    θ  = prev.param
    k  = prev.order

    k     = model_order(model, θ)
    j     = rand(rng, DiscreteUniform(1, k))
    θ′, θⱼ = local_deleteat(model, θ, j)
    k′     = k - 1

    G⁻¹(θ_)   = local_insert(model, θ_, j, θⱼ)
    ϕktok′(θ_) = log(1/k)
    ϕk′tok(θ_) = local_proposal_logpdf(model, proposal.local_proposal, θ, j) + log(1/(k′ + 1))

    θ, ℓπ′, ℓr, stat = propose_ais(rng, proposal, mcmc, model, θ′, ℓπ, G⁻¹, ϕktok′, ϕk′tok)
    RJState(θ, ℓπ′, k′, stat), ℓr
end

