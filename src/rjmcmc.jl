
struct ReversibleBirthDeath{
    Move,
    Jump    <: AbstractJumpProposal
    Sampler <: AbstractMCMC.AbstractSampler
}
    move_kernel  ::Move
    jump_proposal::Jump
    inner_sampler::Sampler
end

function ReversibleBirthDeath(
    order_prior  ::Distributions.DiscreteDistribution,
    jump_proposal::AbstractJumpProposal
)
    move_kernel(k, k′) = 0.25*min(pdf(prior_order, k′)/pdf(prior_order, k), 1)
    Reversible(move_kernel, jump_proposal)
end

function AbstractMCMC.step(
    rng  ::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
         ::ReversibleBirthDeath;
    initial_params,
    kwargs...,
)
    JumpState(
        initial_params, 
        logdensity(model, initial_params),
        model_order(model, initial_params),
        NamedTuple()
    )
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.AbstractModel,
    sampler::ReversibleBirthDeath,
    prev   ::JumpState;
    kwargs...,
)
    @unpack jump_proposal, move_kernel, inner_sampler = sampler

    k  = prev_jump_state.order
    bₖ = move_kernel(k, k+1)
    dₖ = move_kernel(k, k-1)

    rjmcmc_moves = [:birth, :death, :update]
    move_idx     = rand(rng, Categorical(bₖ, dₖ, 1 - (bₖ + dₖ)))
    move         = (k == 0) ? :birth : rjmcmc_moves[move_idx]

    next = step_jump_move(
        rng, move, jump_proposal, inner_sampler, prev, model, move_kernel
    )
    next.param, next
end


