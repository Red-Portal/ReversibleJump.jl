
struct Birth  <: AbstractJumpMove end
struct Death  <: AbstractJumpMove end

function propose_jump(
    rng  ::Random.AbstractRNG,
         ::Birth,
    jump ::IndepJumpProposal,
    prev ::RJState,
         ::Any,
    model,
)
    #= 
        Independent Birth Proposal

        Peter J. Green
        "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination"
        Biometrika, 1995.
    =##
    ℓπ = prev.lp
    θ  = prev.param
    k  = prev.order

    k  = prev.order
    k′  = k + 1
    j  = rand(rng, DiscreteUniform(1, k + 1))

    newborn = local_proposal_sample(rng, model, jump.local_proposal)
    θ′       = local_insert(model, θ, j, newborn)

    ℓφ′ = local_proposal_logpdf(model, jump.local_proposal, θ, j) + log(1/(k + 1))
    ℓφ = log(1/k′)

    ℓπ′ = logdensity(model, θ′)
    RJState(θ′, ℓπ′, k′, NamedTuple()), (ℓπ′ - ℓπ) - (ℓφ′ - ℓφ)
end

function propose_jump(
    rng  ::Random.AbstractRNG,
         ::Death,
    jump ::IndepJumpProposal,
    prev ::RJState,
         ::Any,
    model
)
    #= 
        Independent Death Proposal

        Peter J. Green
        "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination"
        Biometrika, 1995.
    =##
    ℓπ = prev.lp
    θ  = prev.param
    k  = prev.order

    k′  = k - 1
    j  = rand(rng, DiscreteUniform(1, k))

    θ′, _ = local_deleteat(model, θ, j)

    ℓφ′ = log(1/k)
    ℓφ = local_proposal_logpdf(model, jump.local_proposal, θ, j) + log(1/(k′ + 1))

    ℓπ′ = logdensity(model, θ′)
    RJState(θ′, ℓπ′, k′, NamedTuple()), (ℓπ′ - ℓπ) - (ℓφ′ - ℓφ)
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

    θ, ℓπ′, ℓr, stat = step_ais(rng, proposal, mcmc, model, θ′, ℓπ, G⁻¹, ϕktok′, ϕk′tok)
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

    θ, ℓπ′, ℓr, stat = step_ais(rng, proposal, mcmc, model, θ′, ℓπ, G⁻¹, ϕktok′, ϕk′tok)
    RJState(θ, ℓπ′, k′, stat), ℓr
end