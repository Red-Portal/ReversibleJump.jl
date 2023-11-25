
struct RJMCMCTestSampler{RJMCMC <: ReversibleJumpMCMC}
    rjmcmc::RJMCMC
end

function MCMCTesting.markovchain_transition(
    rng::Random.AbstractRNG, model::DiscreteModel, sampler::RJMCMCTestSampler, θ, ::Any
)
    rjmcmc = sampler.rjmcmc
    _, init_state = AbstractMCMC.step(
        rng, model, rjmcmc; initial_params=θ, initial_order=length(θ)
    )
    _, state = AbstractMCMC.step(
        rng, model, rjmcmc, init_state
    )
    state.param
end

@testset "rjmcmc" begin
    rng    = Random.default_rng()
    model  = DiscreteModel(Poisson(4))

    n_anneal = 8
    prop     = ConstantLocalProposal()
    mcmc     = IdentityKernel()

    n_pvalue_samples = 32
    n_rank_samples   = 100
    n_mcmc_steps     = 10
    n_mcmc_thin      = 10
    test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
    statistics       = θ -> [length(θ)]

    @testset for jump in [
        AnnealedJumpProposal(n_anneal, prop, ArithmeticPath()),
        AnnealedJumpProposal(n_anneal, prop, GeometricPath()),
        IndepJumpProposal(prop)
    ]
        rjmcmc  = ReversibleJump.ReversibleJumpMCMC(model.order_dist, jump, mcmc)
        subject = TestSubject(model, RJMCMCTestSampler(rjmcmc))
        @test seqmcmctest(rng, test, subject, 0.001, n_pvalue_samples;
                          statistics, show_progress=false)
    end
end
