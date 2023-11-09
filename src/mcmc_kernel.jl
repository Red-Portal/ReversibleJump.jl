
# function mcmc_step(
#     rng::Random.AbstractRNG,
#     sampler::AbstractMCMC.AbstractSampler,
#     model,
#     θ
# )
#     _, state = AbstractMCMC.step(rng, model, sampler; initial_params=θ)
#     θ′, _     = AbstractMCMC.step(rng, model, sampler, state)
#     θ′
# end
