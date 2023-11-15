
@testset "proposal_indep" begin
    model = CategoricalModelSpace(Categorical([0.2, 0.8]), -30)
    prop  = ConstantLocalProposal()
    jump  = IndepJumpProposal(prop)
    mcmc  = IdentityKernel()

    @testset "birth" begin
        rng    = Random.default_rng()
        θ_init = Float64[]
        state  = ReversibleJump.RJState(
            θ_init, logdensity(model, θ_init), 0, NamedTuple()
        )

        state′, ℓr = ReversibleJump.propose_jump(rng, ReversibleJump.Birth(), jump, state, mcmc, model)

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

        state′, ℓr = ReversibleJump.propose_jump(rng, ReversibleJump.Death(), jump, state, mcmc, model)
        
        logratio_true = logdensity(model, state′.param) - logdensity(model, θ_init) 
        @test ℓr          ≈ logratio_true
        @test state′.lp    ≈ logdensity(model, state′.param)
        @test state′.order ≈ length(state′.param)
    end
end
