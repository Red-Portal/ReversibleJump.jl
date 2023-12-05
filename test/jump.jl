
@testset "jump with perfect mcmc transitions" begin
    prop  = ConstantLocalProposal()
    jump  = IndepJumpProposal(prop)
    mcmc  = IdentityKernel()

    n_samples = 1024

    @testset for jump_proposal = [AnnealedJumpProposal(4, prop, GeometricPath()),
                                  IndepJumpProposal(prop)]
        @testset "birth" begin
            move   = ReversibleJump.Birth()
            rng    = Random.default_rng()
            model  = DiscreteModel(Categorical([0.8, 0.2]))
            θ_init = zeros(1)
            k_init = length(θ_init) 
            state  = ReversibleJump.RJState(
                θ_init, logdensity(model, θ_init), k_init, NamedTuple()
            )

            state′ = ReversibleJump.transition_jump(
                rng, move, jump_proposal, state, mcmc, model, (a,b) -> 1.0
            )
            @assert haskey(state′.stats, :move)           && state′.stats.move           == :jump
            @assert haskey(state′.stats, :jump_move)      && state′.stats.jump_move      == :birth
            @assert haskey(state′.stats, :proposal_order) && state′.stats.proposal_order == k_init + 1
            @assert haskey(state′.stats, :previous_order) && state′.stats.previous_order == k_init
            @assert haskey(state′.stats, :jump_acceptance_rate)

            results = map(1:n_samples) do _
                state′ = ReversibleJump.transition_jump(
                    rng, move, jump_proposal, state, mcmc, model, (a,b) -> 1.0
                )
                param_jumped = state′.order != state.order
                stat_jumps   = state′.stats.jump_accepted
                (param_jumped, stat_jumps)
            end

            order_prob = probs(model.order_dist)
            ratio_true = order_prob[k_init+1]/order_prob[k_init]
            @test all(map(first, results) .== map(last, results))
            @test mean(first, results) ≈ min(ratio_true, 1.0) atol = 0.1
        end

        @testset "death" begin
            move   = ReversibleJump.Death()
            rng    = Random.default_rng()
            model  = DiscreteModel(Categorical([0.2, 0.8]))
            θ_init = zeros(2)
            k_init = length(θ_init) 
            state  = ReversibleJump.RJState(
                θ_init, logdensity(model, θ_init), k_init, NamedTuple()
            )

            state′ = ReversibleJump.transition_jump(
                rng, move, jump_proposal, state, mcmc, model, (a,b) -> 1.0
            )

            @assert haskey(state′.stats, :move)           && state′.stats.move           == :jump
            @assert haskey(state′.stats, :jump_move)      && state′.stats.jump_move      == :death
            @assert haskey(state′.stats, :proposal_order) && state′.stats.proposal_order == k_init - 1
            @assert haskey(state′.stats, :previous_order) && state′.stats.previous_order == k_init
            @assert haskey(state′.stats, :jump_acceptance_rate)
            
            results = map(1:n_samples) do _
                state′ = ReversibleJump.transition_jump(
                    rng, move, jump_proposal, state, mcmc, model, (a,b) -> 1.0
                )
                param_jumped = state′.order != state.order
                stat_jumps   = state′.stats.jump_accepted
                (param_jumped, stat_jumps)
            end

            order_prob = probs(model.order_dist)
            ratio_true = order_prob[k_init-1]/order_prob[k_init]
            @test all(map(first, results) .== map(last, results))
            @test mean(first, results) ≈ min(ratio_true, 1.0) atol = 0.1
        end
    end
end
