
@testset "proposal_ais" begin
    rng   = Random.default_rng()
    model = DiscreteModel(Categorical([0.2, 0.8]))
    prop  = ConstantLocalProposal()
    mcmc  = IdentityKernel()

    @testset for annealing_path = [GeometricPath(4), ArithmeticPath(4)]
        jump  = AnnealedJumpProposal(prop, annealing_path)

        @testset "birth" begin
            rng    = Random.default_rng()
            θ_init = zeros(1)
            state  = ReversibleJump.RJState(
                θ_init, logdensity(model, θ_init), length(θ_init), NamedTuple()
            )

            state′, ℓr = ReversibleJump.propose_jump(rng, ReversibleJump.Birth(), jump, state, mcmc, model)

            logratio_true = logdensity(model, state′.param) - logdensity(model, θ_init) 
            @test ℓr          ≈ logratio_true
            @test state′.lp    ≈ logdensity(model, state′.param)
            @test state′.order ≈ length(state′.param)
        end

        @testset "death" begin
            rng    = Random.default_rng()
            θ_init = zeros(2)
            state  = ReversibleJump.RJState(
                θ_init, logdensity(model, θ_init), length(θ_init), NamedTuple()
            )

            state′, ℓr = ReversibleJump.propose_jump(rng, ReversibleJump.Death(), jump, state, mcmc, model)

            logratio_true = logdensity(model, state′.param) - logdensity(model, θ_init) 
            @test ℓr          ≈ logratio_true
            @test state′.lp    ≈ logdensity(model, state′.param)
            @test state′.order ≈ length(state′.param)
        end
    end
end
