### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ dfffe360-1907-4eb4-9e7d-0b911a61604f
begin
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ b2ead442-899f-11ee-07a6-433b9c11e97d
begin
	using Revise
	using AbstractMCMC
	using SinusoidDetection
	using FillArrays
	using ReversibleJump
	using Random
	using Distributions
	using PDMats
	using Plots
 	plotly()
end


# ╔═╡ 32cf83fc-9a18-4899-b440-56395228dd8a
begin
	rng    = Random.default_rng()
	nu0    = 10.
	gamma0 = 10.
	delta  = 3.
	N      = 32
	model  = SinusoidModel(rng, N, gamma0, nu0, delta)
end

# ╔═╡ e2a669a7-953a-4eeb-a410-6fe5dd092c50
begin
	prior  = model.orderprior
	path   = ArithmeticPath()
	prop   = SinusoidUniformLocalProposal()
	mcmc   = IMHSinusoid(N)
	θ₀     = Float64[]
	T      = 4
	jump   = AnnealedJumpProposal(T, prop, path)
	rjmcmc = ReversibleJumpMCMC(prior, jump, mcmc)

	_, state = AbstractMCMC.step(
    rng, model, rjmcmc; initial_params = θ₀, initial_order = length(θ₀))

	n_samples  = 10^3
    order_hist = Int[]
    for i = 1:10000
        param, state = AbstractMCMC.step(rng, model, rjmcmc, state)
        push!(order_hist, length(param))
    end

	plot(order_hist, ylabel="k", xlabel="RJMCMC Iteration", label="k")
end

# ╔═╡ c05195bd-47ef-45f7-a651-6a6c16337821


# ╔═╡ Cell order:
# ╠═dfffe360-1907-4eb4-9e7d-0b911a61604f
# ╠═b2ead442-899f-11ee-07a6-433b9c11e97d
# ╠═32cf83fc-9a18-4899-b440-56395228dd8a
# ╠═e2a669a7-953a-4eeb-a410-6fe5dd092c50
# ╠═c05195bd-47ef-45f7-a651-6a6c16337821
