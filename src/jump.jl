
function step_jump(
    rng     ::Random.AbstractRNG,
    move    ::AbstractJumpMove,
    proposal::AbstractJumpProposal,
    prev    ::RJState,
    mcmc,
    model,
    move_kernel,
)
    prop, ℓr = propose_jump(rng, move, proposal, prev, mcmc, model)
    k′, k     = prop.order, prev.order 
    q_k′, q_k = move_kernel(k, k′), move_kernel(k′, k)

    ℓα = min(0, ℓr - log(q_k′/q_k))
    α  = exp(ℓα)
        
    stats = (move = move, jump_acceptance_rate=α,)
    
    if  log(rand(rng)) < ℓα
        stats′ = merge(stats, (jump_accepted=true,))
        @set prop.stats = stats′
    else
        stats′ = merge(stats, (jump_accepted=false,))
        @set prev.stats = stats′
    end
end
