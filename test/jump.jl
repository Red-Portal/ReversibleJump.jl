
@testset "jump" begin
    prop  = ConstantLocalProposal()
    jump  = IndepJumpProposal(prop)
    mcmc  = IdentityKernel()

    n_samples = 1024

    @testset for jump_proposal = [AnnealedJumpProposal(4, prop, GeometricPath()),
                                  IndepJumpProposal(prop)]
        @testset "birth" begin
            move   = ReversibleJump.Birth()
            rng    = Random.default_rng()
            model  = CategoricalModelSpace(Categorical([0.8, 0.2]), -30)
            θ_init = Float64[]
            state  = ReversibleJump.RJState(
                θ_init, logdensity(model, θ_init), 0, NamedTuple()
            )

            results = map(1:n_samples) do _
                state′ = ReversibleJump.transition_jump(rng, move, jump_proposal, state, mcmc, model, (a,b) -> 1.0)
                (state′.order != state.order, state′.stats.jump_accepted)
            end

            ratio_true = 0.2/0.8
            @test all(map(first, results) .== map(last, results))
            @test mean(first, results) ≈ min(ratio_true, 1.0) atol = 0.1
        end

        @testset "death" begin
            move   = ReversibleJump.Death()
            rng    = Random.default_rng()
            model  = CategoricalModelSpace(Categorical([0.2, 0.8]), -30)
            θ_init = [1.0]
            state  = ReversibleJump.RJState(
                θ_init, logdensity(model, θ_init), 1, NamedTuple()
            )
            
            results = map(1:n_samples) do _
                state′ = ReversibleJump.transition_jump(rng, move, jump_proposal, state, mcmc, model, (a,b) -> 1.0)
                (state′.order != state.order, state′.stats.jump_accepted)
            end

            ratio_true = 0.2/0.8
            @test all(map(first, results) .== map(last, results))
            @test mean(first, results) ≈ min(ratio_true, 1.0) atol = 0.1
        end
    end
end
