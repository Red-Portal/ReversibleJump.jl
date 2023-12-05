
function transition_jump(
    rng     ::Random.AbstractRNG,
    move    ::AbstractJumpMove,
    proposal::AbstractJumpProposal,
    prev    ::AbstractRJState,
    mcmc,
    model,
    order_kernel,
)
    prop, ℓr     = propose_jump(rng, move, proposal, prev, mcmc, model)
    k′, k         = prop.order, prev.order 
    qktok′, qk′tok = order_kernel(k, k′), order_kernel(k′, k)

    ℓα = min(0, ℓr - log(qktok′/qk′tok))
    α  = exp(ℓα)
        
    stats = (move                 = :jump,
             jump_move            = Symbol(move),
             previous_order       = k,
             proposal_order       = k′,
             jump_acceptance_rate = α,)
    
    if  log(rand(rng)) < ℓα
        stats′ = merge(stats, (jump_accepted=true,))
        @set prop.stats = stats′
    else
        stats′ = merge(stats, (jump_accepted=false,))
        @set prev.stats = stats′
    end
end
