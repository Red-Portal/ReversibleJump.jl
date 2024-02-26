
struct ReversibleJumpMCMC{
    JumpProp   <: AbstractJumpProposal,
    MovePairs  <: Vector{<:AbstractJumpMovePair},
    MoveWeight <: StatsBase.AbstractWeights,
    MoveKern,
    MCMCKern,
} <: AbstractRJMCMCSampler
    jump_proposal::JumpProp
    move_pairs   ::MovePairs
    move_weights ::MoveWeight
    order_kernel ::MoveKern
    mcmc_kernel  ::MCMCKern
end

function ReversibleJumpMCMC(
    order_prior  ::Distributions.DiscreteDistribution,
    jump_proposal::AbstractJumpProposal,
    mcmc_kernel,
    move_pairs   ::AbstractVector{<:AbstractJumpMovePair} = [BirthDeath()],
    move_weights ::StatsBase.AbstractWeights              = pweights(fill(1.0, length(move_pairs)));
    jump_rate    ::Real                                   = 0.5, 
)
    @assert length(move_weights) == length(move_pairs)
    ϵ = eps(typeof(jump_rate))
    order_kernel(k, k′) = jump_rate*min(pdf(order_prior, k′)/(pdf(order_prior, k) + ϵ), 1)
    ReversibleJumpMCMC(
        jump_proposal, move_pairs, move_weights, order_kernel, mcmc_kernel
    )
end

function ReversibleJumpMCMC(
    jump_proposal::AbstractJumpProposal,
    mcmc_kernel,
    order_kernel,
    move_pairs   ::AbstractVector{<:AbstractJumpMovePair} = [BirthDeath()],
    move_prob    ::StatsBase.AbstractWeights              = pweights(fill(1.0, length(move_pairs))),
)
    @assert length(move_prob) == length(move_pairs)
    ReversibleJumpMCMC(
        jump_proposal, move_pairs, move_prob, order_kernel, mcmc_kernel
    )
end

function AbstractMCMC.step(
    rng  ::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
         ::ReversibleJumpMCMC;
    initial_params,
    initial_order,
    kwargs...,
)
    initial_params, RJState(
        initial_params, 
        logdensity(model, initial_params),
        initial_order,
        NamedTuple()
    )
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.AbstractModel,
    sampler::ReversibleJumpMCMC,
    prev   ::RJState;
    kwargs...,
)
    @unpack jump_proposal, move_pairs, move_weights, order_kernel, mcmc_kernel = sampler

    move_pair       = StatsBase.sample(rng, move_pairs, move_weights)
    k               = prev.order
    move_fwd, k_fwd = forward_move(move_pair, k)
    move_bwd, k_bwd = backward_move(move_pair, k)
    p_fwd           = order_kernel(k, k_fwd)
    p_bwd           = order_kernel(k, k_bwd)
    p_update        = 1 - (p_fwd + p_bwd)
    stats           = (update_rate = p_update,)

    next = if rand(rng, Bernoulli(p_update))
        next_param, lp = transition_mcmc(rng, mcmc_kernel, model, prev.param)
        RJState(next_param, lp, prev.order, (move = :update,))
    else
        move = rand(rng, Bernoulli(p_fwd/(p_fwd + p_bwd))) ? move_fwd : move_bwd
        transition_jump(
            rng, move, jump_proposal, prev, mcmc_kernel, model, order_kernel
        )       
    end
    next = RJState(next.param, next.lp, next.order, merge(stats, next.stats))
    next.param, next
end

