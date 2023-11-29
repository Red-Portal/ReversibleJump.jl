
function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::SinusoidFixedOrderModel{<:SinusoidKnownSNR},
    mcmc ::IMHRWMHKnownSNR,
    θ, y
)
    model_base = model.model
    model_base = @set model_base.y = y
    ReversibleJump.transition_mcmc(rng, mcmc, model_base, copy(θ)) |> first
end

function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::SinusoidFixedOrderModel{<:SinusoidUnknownSNR},
    mcmc ::IMHRWMHUnknownSNR,
    θ, y
)
    model_base = model.model
    model_base = @set model_base.y = y
    ReversibleJump.transition_mcmc(rng, mcmc, model_base, copy(θ)) |> first
end

@testset "imhrwmh known snr" begin
    nu0    = 2.0
    gamma0 = 5.0
    delta2 = 8.0
    N      = 16

    k          = 1
    model_base = rand_sinusoids_knownsnr(N, nu0, gamma0, delta2)
    model      = SinusoidFixedOrderModel(k, model_base)
    _, y       = MCMCTesting.sample_joint(Random.default_rng(), model)
    model      = @set model.model.y = y

    n_pvalue_samples = 32
    n_samples        = 100
    n_mcmc_steps     = 10
    test             = TwoSampleTest(n_samples, n_mcmc_steps)

    mcmc    = IMHRWMHKnownSNR(Uniform(0, π), N)
    subject = TestSubject(model, mcmc)
    @test seqmcmctest(test, subject, 0.0001, n_pvalue_samples; show_progress=true)
end

@testset "imhrwmh unknown snr" begin
    ν0   = 2.0
    γ0   = 5.0
    α_δ² = 2.0
    β_δ² = 5.0
    N    = 16

    k          = 1
    model_base = rand_sinusoids_unknownsnr(N, ν0, γ0, α_δ², β_δ²)
    model      = SinusoidFixedOrderModel(k, model_base)
    _, y       = MCMCTesting.sample_joint(Random.default_rng(), model)
    model      = @set model.model.y = y

    n_pvalue_samples = 32
    n_samples        = 100
    n_mcmc_steps     = 10
    test             = TwoSampleTest(n_samples, n_mcmc_steps)

    mcmc    = IMHRWMHUnknownSNR(Uniform(0, π), N)
    subject = TestSubject(model, mcmc)
    @test seqmcmctest(test, subject, 0.0001, n_pvalue_samples; show_progress=true)
end
