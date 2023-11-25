
struct NRJMCMCTestSampler{NRJMCMC <: NonReversibleJumpMCMC}
    nrjmcmc::NRJMCMC
end

function MCMCTesting.markovchain_transition(
    rng::Random.AbstractRNG, model::DiscreteModel, sampler::NRJMCMCTestSampler, θ, ::Any
)
    nrjmcmc = sampler.nrjmcmc
    _, init_state = AbstractMCMC.step(
        rng, model, nrjmcmc; initial_params=θ, initial_order=length(θ)
    )
    param, _ = AbstractMCMC.step(
        rng, model, nrjmcmc, init_state
    )
    param
end

@testset "nrjmcmc" begin
    rng    = Random.default_rng()
    model  = DiscreteModel(Poisson(4))

    n_anneal = 8
    prop     = ConstantLocalProposal()
    mcmc     = IdentityKernel()

    n_pvalue_samples = 32
    n_samples        = 100
    n_mcmc_steps     = 10
    test             = TwoSampleTest(n_samples, n_samples)
    statistics       = θ -> [length(θ)]

    @testset for jump in [
        AnnealedJumpProposal(n_anneal, prop, ArithmeticPath()),
        AnnealedJumpProposal(n_anneal, prop, GeometricPath()),
        IndepJumpProposal(prop)
    ]
        nrjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc)
        subject = TestSubject(model, NRJMCMCTestSampler(nrjmcmc))
        @test seqmcmctest(rng, test, subject, 0.001, n_pvalue_samples;
                          statistics, show_progress=false)
    end
end
