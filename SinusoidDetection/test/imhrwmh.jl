
function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::SinusoidFixedOrderModel,
    mcmc ::IMHRWMHSinusoid,
    θ, y
)
    model_base = model.model
    model_base = @set model_base.y = y
    ReversibleJump.transition_mcmc(rng, mcmc, model_base, copy(θ)) |> first
end

@testset "imhrwmh" begin
    nu0    = 10.0
    gamma0 = 10.0
    delta  = 8.0
    N      = 16

    k          = 2
    model_base = rand_sinusoids(N, gamma0, nu0, delta)
    model      = SinusoidFixedOrderModel(k, model_base)
    _, y       = MCMCTesting.sample_joint(Random.default_rng(), model)
    model      = @set model.model.y = y

    n_pvalue_samples = 32
    n_rank_samples   = 100
    n_mcmc_steps     = 10
    n_mcmc_thin      = 1
    test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)

    mcmc    = IMHRWMHSinusoid(N)
    subject = TestSubject(model, mcmc)
    @test seqmcmctest(test, subject, 0.001, n_pvalue_samples; show_progress=true)
end
