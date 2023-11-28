### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ dfffe360-1907-4eb4-9e7d-0b911a61604f
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ b2ead442-899f-11ee-07a6-433b9c11e97d
begin
	using AbstractMCMC
	using Accessors
	using Distributions
	using Revise
	using SinusoidDetection
	using SimpleUnPack
	using ReversibleJump
	using Random
	using MCMCChains
	using MCMCTesting
	
	using Plots, StatsPlots
	gr()
end


# ╔═╡ 928b553b-fa4d-462f-adf8-37605ff9918c
begin
struct SinusoidFixedOrderModel{Model <: SinusoidDetection.AbstractSinusoidModel}
    k::Int
    model::Model
end

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::SinusoidKnownSNR)
    @unpack y, nu0, gamma0, delta, orderprior = model

    N  = length(y)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = delta*delta
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², δ²)
    ω, y
end

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::SinusoidUnknownSNR)
    @unpack y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior = model

    N  = length(y)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = rand(rng, InverseGamma(alpha_delta2, beta_delta2))
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², δ²)
    θ  = vcat([δ²], ω)
    θ, y
end

function MCMCTesting.sample_joint(
    rng::Random.AbstractRNG, model::SinusoidFixedOrderModel{<:SinusoidKnownSNR}
)
    @unpack y, gamma0, nu0, delta, orderprior = model.model

    N  = length(y)
    k  = model.k
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = delta*delta
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², δ²)
    ω, y
end

function MCMCTesting.sample_joint(
    rng::Random.AbstractRNG, model::SinusoidFixedOrderModel{<:SinusoidUnknownSNR}
)
    @unpack y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior = model.model

    N  = length(y)
    k  = model.k
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = rand(rng, InverseGamma(alpha_delta2, beta_delta2))
    y  = SinusoidDetection.sample_signal(rng, ω, N, σ², δ²)
    θ  = vcat([δ²], ω)
    θ, y
end

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
    param, _ = AbstractMCMC.step(
        rng, model, rjmcmc, init_state
    )
    param
end

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

function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::SinusoidFixedOrderModel{<:SinusoidKnownSNR},
    mcmc ::IMHRWMHSinusoidKnownSNR,
    θ, y
)
    model_base = model.model
    model_base = @set model_base.y = y
    ReversibleJump.transition_mcmc(rng, mcmc, model_base, copy(θ)) |> first
end

function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::SinusoidFixedOrderModel{<:SinusoidUnknownSNR},
    mcmc ::IMHRWMHSinusoidUnknownSNR,
    θ, y
)
    model_base = model.model
    model_base = @set model_base.y = y
    ReversibleJump.transition_mcmc(rng, mcmc, model_base, copy(θ)) |> first
end
end

# ╔═╡ e2a669a7-953a-4eeb-a410-6fe5dd092c50
begin
	ν0    = 2.0
    γ0    = 5.0
    δ     = 8.0
    N     = 16
    α_δ²  = 5.0
    β_δ²  = 10.0
    model = rand_sinusoids_unknownsnr(N, ν0, γ0, α_δ², β_δ²)

    prior = Geometric(0.2)
    path  = ArithmeticPath()
    prop  = SinusoidUniformLocalProposal()
    mcmc  = IMHRWMHSinusoidUnknownSNR(N)
	#mcmc  = SliceSteppingOut(2.0)

    jump   = IndepJumpProposal(prop)
    rjmcmc = ReversibleJumpMCMC(prior, jump, mcmc)

    n_samples    = 3000
    n_mcmc_steps = 30
    n_mcmc_thin  = 1
    test         = ExactRankTest(n_samples, n_mcmc_steps, n_mcmc_thin)
    statistics   = θ -> Float64[θ[1], θ[1]^2, length(θ), θ[1]*length(θ)]

	subject = TestSubject(model, rjmcmc)
end

# ╔═╡ 5a2d138f-1312-4b4c-b270-f87b3215abb2
ranks = simulate_ranks(test, subject; statistics, show_progress=false)


# ╔═╡ 3cdea5fa-a96d-4d39-af40-6b03bd4593c2
rankplot(test, ranks)

# ╔═╡ Cell order:
# ╠═dfffe360-1907-4eb4-9e7d-0b911a61604f
# ╠═b2ead442-899f-11ee-07a6-433b9c11e97d
# ╠═928b553b-fa4d-462f-adf8-37605ff9918c
# ╠═e2a669a7-953a-4eeb-a410-6fe5dd092c50
# ╠═5a2d138f-1312-4b4c-b270-f87b3215abb2
# ╠═3cdea5fa-a96d-4d39-af40-6b03bd4593c2
