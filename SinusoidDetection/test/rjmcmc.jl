
function MCMCTesting.markovchain_transition(
    rng::Random.AbstractRNG, model::SinusoidModel, rjmcmc::ReversibleJumpMCMC, θ, y
)
    model  = @set model.y = y
    _, init_state = AbstractMCMC.step(
        rng, model, rjmcmc; initial_params=θ, initial_order=length(θ)
    )
    _, state = AbstractMCMC.step(
        rng, model, rjmcmc, init_state
    )
    state.param
end

@testset "rjmcmc" begin
    nu0    = 10.0
    gamma0 = 10.0
    delta  = 8.0
    N      = 16
    model  = rand_sinusoids(N, gamma0, nu0, delta)

    prior = Geometric(0.2)
    path  = ArithmeticPath()
    prop  = SinusoidUniformLocalProposal()
    mcmc  = IMHRWMHSinusoid(N)

    T      = 4
    jump   = AnnealedJumpProposal(T, prop, path)
    rjmcmc = ReversibleJumpMCMC(prior, jump, mcmc)

    n_pvalue_samples = 32
    n_rank_samples   = 100
    n_mcmc_steps     = 10
    n_mcmc_thin      = 1
    test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
    statistics       = θ -> [length(θ)]

    rjmcmc  = ReversibleJump.ReversibleJumpMCMC(prior, jump, mcmc)
    subject = TestSubject(model, rjmcmc)
    @test seqmcmctest(test, subject, 0.001, n_pvalue_samples;
                      statistics, show_progress=true)
end
