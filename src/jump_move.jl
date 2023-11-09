
struct JumpState{Param, NT <: NamedTuple}
    param::Param
    lp   ::Real
    order::Int
    stat ::NT
end

function step_jump_move(
    rng          ::Random.AbstractRNG,
    move         ::Symbol,
    jump         ::AbstractJumpProposal,
    inner_sampler::AbstractMCMC.AbstractSampler,
    prev         ::JumpState,
    model,
    move_kernel,
)
    #=
        Birth-Death Reversible Jump Annealed Importance Sampling Kernel

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible Jump MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.

        Birth-death type moves were orginally proposed in

        Peter J. Green
        "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination"
        Biometrika, 1995.
    =##

    if move == :birth || move == :death
        proposal, ℓr = if move == :birth
            proposal_birth(rng, jump, inner_sampler, prev, model)
        else
            proposal_death(rng, jump, inner_sampler, prev, model)
        end

        k′, k     = prop.order, prev.order 
        q_k′, q_k = move_kernel(k, k′), move_kernel(k′, k)

        ℓα = min(0, ℓr - log(q_k′/q_k))
        α  = exp(ℓα)
        
        stats = merge(stats, (move = move, jump_rate=α,))

        if  log(rand(rng)) < ℓα
            stat′ = merge(stats, (has_jumped=true,))
            @set prop.stat = stat′
        else
            stat′ = merge(stats, (has_jumped=false,))
            @set prev.stat = stat′
        end
    else
        single_transition(rng, inner_sampler, model, prev)
    end
end
