
struct Birth      <: AbstractJumpMove     end
struct Death      <: AbstractJumpMove     end
struct BirthDeath <: AbstractJumpMovePair end

forward_move( ::BirthDeath, k) = (Birth(), k+1)
backward_move(::BirthDeath, k) = (Death(), k-1)

Base.show(io::IO, ::Birth) = print(io, "birth")
Base.show(io::IO, ::Death) = print(io, "death")

function propose_jump(
    rng  ::Random.AbstractRNG,
         ::Birth,
    jump ::IndepJumpProposal,
    prev ::AbstractRJState,
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
    k′  = k + 1

    j       = rand(rng, DiscreteUniform(1, k + 1))
    newborn = local_proposal_sample(rng, model, jump.local_proposal)
    θ′       = local_insert(model, θ, j, newborn)

    ℓq_fwd = local_proposal_logpdf(model, jump.local_proposal, θ′, j) + log(1/k′)
    ℓq_bwd = log(1/k′)

    ℓπ′  = logdensity(model, θ′)
    prop = setproperties(prev, (param = θ′,
                                lp    = ℓπ′,
                                order = k′,
                                stats = NamedTuple()))
    prop, (ℓπ′ - ℓπ) - (ℓq_fwd - ℓq_bwd)
end

function propose_jump(
    rng  ::Random.AbstractRNG,
         ::Death,
    jump ::IndepJumpProposal,
    prev ::AbstractRJState,
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

    ℓq_fwd = log(1/k)
    ℓq_bwd = local_proposal_logpdf(model, jump.local_proposal, θ, j) + log(1/k)

    ℓπ′ = logdensity(model, θ′)
    prop = setproperties(prev, (param = θ′,
                                lp    = ℓπ′,
                                order = k′,
                                stats = NamedTuple()))
    prop, (ℓπ′ - ℓπ) - (ℓq_fwd - ℓq_bwd)
end

function propose_jump(
    rng     ::Random.AbstractRNG,
            ::Birth,
    proposal::AnnealedJumpProposal,
    prev    ::AbstractRJState,
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

    map_fwd(θ_) = θ_
    map_bwd(θ_) = first(local_deleteat(model, θ_, j))
    ℓq_fwd(θ_)  = local_proposal_logpdf(model, proposal.local_proposal, θ_, j) + log(1/k′)
    ℓq_bwd(θ_)  = log(1/k′)

    θ′, ℓπ′, ℓr, stat = step_ais(
        rng, proposal, mcmc, model, θ′, ℓπ, map_fwd, map_bwd, ℓq_fwd, ℓq_bwd
    )
    prop = setproperties(prev, (param = map_fwd(θ′),
                                lp    = ℓπ′,
                                order = k′,
                                stats = stat))
    prop, ℓr
end

function propose_jump(
    rng     ::Random.AbstractRNG,
            ::Death,
    proposal::AnnealedJumpProposal,
    prev    ::AbstractRJState,
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

    j = rand(rng, DiscreteUniform(1, k))
    k′ = k - 1

    map_fwd(θ_) = first(local_deleteat(model, θ_, j))
    map_bwd(θ_) = θ_
    ℓq_fwd(θ_)  = log(1/k)
    ℓq_bwd(θ_)  = local_proposal_logpdf(model, proposal.local_proposal, θ_, j) + log(1/k)

    θ′, ℓπ′, ℓr, stat = step_ais(
        rng, proposal, mcmc, model, θ, ℓπ, map_fwd, map_bwd, ℓq_fwd, ℓq_bwd
    )
    prop = setproperties(prev, (param = map_fwd(θ′),
                                lp    = ℓπ′,
                                order = k′,
                                stats = stat))
    prop, ℓr
end
