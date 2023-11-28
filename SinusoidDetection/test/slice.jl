
function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::SinusoidFixedOrderModel,
    mcmc ::SinusoidDetection.AbstractSliceSampling,
    θ, y
)
    model_base = model.model
    model_base = @set model_base.y = y
    ReversibleJump.transition_mcmc(rng, mcmc, model_base, copy(θ)) |> first
end

@testset "slice samplers known snr" begin
    ν0 = 2.0
    γ0 = 5.0
    δ² = 8.0
    N  = 16

    k          = 2
    model_base = rand_sinusoids_knownsnr(N, ν0, γ0, δ²)
    model      = SinusoidFixedOrderModel(k, model_base)
    _, y       = MCMCTesting.sample_joint(Random.default_rng(), model)
    model      = @set model.model.y = y

    n_pvalue_samples = 32
    n_rank_samples   = 100
    n_mcmc_steps     = 10
    n_mcmc_thin      = 1
    test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
    
    window = 2.0
    for mcmc in [
        Slice(window),
        SliceDoublingOut(window),
        SliceSteppingOut(window)
    ]
        subject = TestSubject(model, mcmc)
        @test seqmcmctest(test, subject, 0.001, n_pvalue_samples; show_progress=true)
    end
end

@testset "slice samplers unknown snr" begin
    ν0   = 2.0
    γ0   = 5.0
    α_δ² = 2.0
    β_δ² = 5.0
    N    = 16

    k          = 2
    model_base = rand_sinusoids_unknownsnr(N, ν0, γ0, α_δ², β_δ²)
    model      = SinusoidFixedOrderModel(k, model_base)
    _, y       = MCMCTesting.sample_joint(Random.default_rng(), model)
    model      = @set model.model.y = y

    n_pvalue_samples = 32
    n_rank_samples   = 100
    n_mcmc_steps     = 10
    n_mcmc_thin      = 1
    test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
    
    window = 2.0
    for mcmc in [
        Slice(window),
        SliceDoublingOut(window),
        SliceSteppingOut(window)
    ]
        subject = TestSubject(model, mcmc)
        @test seqmcmctest(test, subject, 0.001, n_pvalue_samples; show_progress=true)
    end
end
