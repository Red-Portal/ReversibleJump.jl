
struct NonReversibleJumpMCMC{
    JumpProp   <: AbstractJumpProposal,
    MovePairs  <: Vector{<:AbstractJumpMovePair},
    MoveWeight <: StatsBase.AbstractWeights,
    UpRate     <: Real, 
    MCMCKern,
} <: AbstractRJMCMCSampler
    jump_proposal::JumpProp
    move_pairs   ::MovePairs
    move_weights ::MoveWeight
    mcmc_kernel  ::MCMCKern
    update_rate  ::UpRate
end

function NonReversibleJumpMCMC(
    jump_proposal::AbstractJumpProposal,
    mcmc_kernel,
    move_pairs   ::AbstractVector{<:AbstractJumpMovePair} = [BirthDeath()],
    move_weights ::StatsBase.AbstractWeights              = pweights(fill(1.0, length(move_pairs)));
    jump_rate    ::Real                                   = 0.5, 
)
    @assert length(move_weights) == length(move_pairs)
    NonReversibleJumpMCMC(
        jump_proposal, move_pairs, move_weights, mcmc_kernel, 1 - jump_rate
    )
end

function AbstractMCMC.step(
    rng  ::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
         ::NonReversibleJumpMCMC;
    initial_params,
    initial_order,
    kwargs...,
)
    initial_params, NRJState(
        rand(rng) > 0.5,
        initial_params, 
        logdensity(model, initial_params),
        initial_order,
        NamedTuple()
    )
    
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.AbstractModel,
    sampler::NonReversibleJumpMCMC,
    prev   ::NRJState;
    kwargs...,
)
    @unpack jump_proposal, move_pairs, move_weights, mcmc_kernel, update_rate = sampler

    move_pair   = StatsBase.sample(rng, move_pairs, move_weights)
    k           = prev.order
    move_fwd, _ = forward_move(move_pair, k)
    move_bwd, _ = backward_move(move_pair, k)
    direction   = prev.direction

    next = if rand(rng) ≤ update_rate
        next_param, lp = transition_mcmc(rng, mcmc_kernel, model, prev.param)
        setproperties(prev, (param = next_param,
                             lp    = lp,
                             stats = (move = :update,)))
    else
        move = if direction || k == 0
            move_fwd
        else
            move_bwd
        end

        next = transition_jump(
            rng, move, jump_proposal, prev, mcmc_kernel, model, (k′, k′′) -> 1.0
        )       
        if !next.stats.jump_accepted
            @set next.direction = !direction
        else
            next
        end
    end
    next.param, next
end
