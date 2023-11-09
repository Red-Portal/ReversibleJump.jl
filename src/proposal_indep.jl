
struct IndepJumpProposal{Prop} <: AbstractJumpProposal
    local_proposal::Prop
end

function proposal_death(
    rng  ::Random.AbstractRNG,
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

    θ′  = local_deleteat(model, θ, j)

    ℓφ′ = log(1/k)
    ℓφ = local_proposal_logpdf(model, jump.local_proposal, θ, j) + log(1/(k′ + 1))

    ℓπ′ = logdensity(model, θ′)
    RJState(θ′, ℓπ′, k′, NamedTuple()), (ℓπ′ - ℓπ) - (ℓφ′ - ℓφ)
end

function proposal_birth(
    rng  ::Random.AbstractRNG,
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

