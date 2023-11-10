
@testset "proposal_ais" begin
    rng   = Random.default_rng()
    model = CategoricalModelSpace(Categorical([0.2, 0.8]), -30)
    prop  = ConstantLocalProposal()

    @testset for annealing_path = [GeometricPath(), ArithmeticPath()]
        jump  = AnnealedJumpProposal(32, prop, annealing_path)
        mcmc  = IdentityKernel()

        @testset "birth" begin
            rng    = Random.default_rng()
            θ_init = Float64[]
            state  = ReversibleJump.RJState(
                θ_init, logdensity(model, θ_init), 0, NamedTuple()
            )

            state′, ℓr = ReversibleJump.propose_jump(rng, Birth(), jump, state, mcmc, model)

            logratio_true = logdensity(model, state′.param) - logdensity(model, θ_init) 
            @test ℓr          ≈ logratio_true
            @test state′.lp    ≈ logdensity(model, state′.param)
            @test state′.order ≈ length(state′.param)
        end

        @testset "death" begin
            rng    = Random.default_rng()
            θ_init = [1.0]
            state  = ReversibleJump.RJState(
                θ_init, logdensity(model, θ_init), 1, NamedTuple()
            )

            state′, ℓr = ReversibleJump.propose_jump(rng, Death(), jump, state, mcmc, model)

            logratio_true = logdensity(model, state′.param) - logdensity(model, θ_init) 
            @test ℓr          ≈ logratio_true
            @test state′.lp    ≈ logdensity(model, state′.param)
            @test state′.order ≈ length(state′.param)
        end
    end
end
