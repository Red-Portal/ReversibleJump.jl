
function MCMCTesting.markovchain_transition(
    rng   ::Random.AbstractRNG,
    model ::SinusoidKnownSNR,
    rjmcmc::ReversibleJumpMCMC,
    θ, y
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

function MCMCTesting.markovchain_transition(
    rng   ::Random.AbstractRNG,
    model ::SinusoidUnknownSNR,
    rjmcmc::ReversibleJumpMCMC,
    θ, y
)
    model  = @set model.y = y
    _, init_state = AbstractMCMC.step(
        rng, model, rjmcmc; initial_params=θ, initial_order=length(θ) - 1
    )
    _, state = AbstractMCMC.step(
        rng, model, rjmcmc, init_state
    )
    state.param
end

@testset "rjmcmc known snr" begin
    ν0    = 2.0
    γ0    = 5.0
    δ     = 8.0
    N     = 16
    model = rand_sinusoids_knownsnr(N, ν0, γ0, δ)

    prior = Geometric(0.2)
    path  = ArithmeticPath()
    prop  = SinusoidUniformLocalProposal()
    mcmc  = IMHRWMHKnownSNR(Uniform(0, π), N)

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

@testset "rjmcmc unknown SNR" begin
    ν0    = 2.0
    γ0    = 5.0
    δ     = 8.0
    N     = 16
    α_δ²  = 2.0
    β_δ²  = 5.0
    model = rand_sinusoids_unknownsnr(N, ν0, γ0, α_δ², β_δ²)

    prior = Geometric(0.2)
    path  = ArithmeticPath()
    prop  = SinusoidUniformLocalProposal()
    mcmc  = IMHRWMHUnknownSNR(Uniform(0, π), N)

    jump   = IndepJumpProposal(prop)
    rjmcmc = ReversibleJumpMCMC(prior, jump, mcmc)

    n_pvalue_samples = 32
    n_rank_samples   = 100
    n_mcmc_steps     = 10
    n_mcmc_thin      = 1
    test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
    statistics       = θ -> [θ[1], length(θ)]

    rjmcmc  = ReversibleJump.ReversibleJumpMCMC(prior, jump, mcmc)
    subject = TestSubject(model, rjmcmc)
    @test seqmcmctest(test, subject, 0.001, n_pvalue_samples;
                      statistics, show_progress=true)
end
