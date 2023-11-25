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
	using Distributions
	using Revise
	using SinusoidDetection
	using ReversibleJump
	using Random
	using Plots, StatsPlots
	using MCMCChains
 	plotly()
end


# ╔═╡ 32cf83fc-9a18-4899-b440-56395228dd8a
begin
	rng    = Random.default_rng()
	nu0    = 10.
	gamma0 = 10.
	delta  = 8.
	N      = 32
	model  = SinusoidModel(rng, N, gamma0, nu0, delta)
end

# ╔═╡ e2a669a7-953a-4eeb-a410-6fe5dd092c50
begin
	prior     = Geometric(0.2)
	path_arit = ArithmeticPath()
    path_geom = GeometricPath()
	prop      = SinusoidUniformLocalProposal()
	mcmc      = IMHSinusoid(N)

    n_samples      = 5000
	initial_params = Float64[]
	initial_order  = 0

	T                = 8
	ais_arit_jump    = AnnealedJumpProposal(T, prop, path_arit)
	ais_geom_jump    = AnnealedJumpProposal(T, prop, path_geom)
	rjmcmc_ais_arit  = ReversibleJumpMCMC(prior, ais_arit_jump, mcmc)
	rjmcmc_ais_geom  = ReversibleJumpMCMC(prior, ais_geom_jump, mcmc)
	nrjmcmc_ais_arit = NonReversibleJumpMCMC(ais_arit_jump, mcmc; jump_rate=0.8)
	nrjmcmc_ais_geom = NonReversibleJumpMCMC(ais_geom_jump, mcmc; jump_rate=0.8)
end


# ╔═╡ 2f234dbd-f8e7-42b9-8c79-2fa01b017380
_, rjmcmc_ais_arit_stats = ReversibleJump.sample(
    rng, rjmcmc_ais_arit, model, n_samples, initial_order, initial_params; show_progress=false
);

# ╔═╡ f2fe90e0-5423-4c5b-b81d-1a5a77c6a0c4
_, rjmcmc_ais_geom_stats = ReversibleJump.sample(
    rng, rjmcmc_ais_geom, model, n_samples, initial_order, initial_params; show_progress=false
);

# ╔═╡ a0198327-8cb2-42e0-88a6-9a9e7e720e19
_, nrjmcmc_ais_arit_stats = ReversibleJump.sample(
	rng, nrjmcmc_ais_arit, model, n_samples, initial_order, initial_params; show_progress=false
);

# ╔═╡ 66f95076-eaa9-4bac-bb93-4b11a8cde7bd
_, nrjmcmc_ais_geom_stats = ReversibleJump.sample(
	rng, nrjmcmc_ais_geom, model, n_samples, initial_order, initial_params; show_progress=false
);

# ╔═╡ 258e721b-6d4a-45e0-9d4b-d5cf2271672b
begin
	chain_rjmcmc_ais_arit = MCMCChains.Chains(
		reshape([stat.order for stat in rjmcmc_ais_arit_stats[n_samples÷2:end]], (:,1,1))
	)
	ess(chain_rjmcmc_ais_arit)
end

# ╔═╡ cbb941e2-379f-4f7e-9bdd-d738024bfed7
begin
	chain_rjmcmc_ais_geom = MCMCChains.Chains(
		reshape([stat.order for stat in rjmcmc_ais_geom_stats[n_samples÷2:end]], (:,1,1))
	)
	ess(chain_rjmcmc_ais_geom)
end

# ╔═╡ 4b5faec2-6570-4933-a2b4-4a986829468f
begin
	chain_nrjmcmc_ais_arit = MCMCChains.Chains(
		reshape([stat.order for stat in nrjmcmc_ais_arit_stats[n_samples÷2:end]], (:,1,1))
	)
	ess(chain_nrjmcmc_ais_arit)
end

# ╔═╡ e563fc61-282b-4252-933a-62d8232e23f5
begin
	chain_nrjmcmc_ais_geom = MCMCChains.Chains(
		reshape([stat.order for stat in nrjmcmc_ais_geom_stats[n_samples÷2:end]], (:,1,1))
	)
	ess(chain_nrjmcmc_ais_geom)
end

# ╔═╡ 7453bd71-9e16-4c1b-8fc6-e5a02f93aed8
traceplot(chain_rjmcmc_ais_arit, title="RJMCMC-AIS-Arithmetic")

# ╔═╡ e48e4872-9e22-4214-b213-26402e3ea19e
traceplot(chain_rjmcmc_ais_geom, title="RJMCMC-AIS-Geometric")

# ╔═╡ 45f35f82-e92c-4f1e-9b4f-908eabf3eae5
traceplot(chain_nrjmcmc_ais_arit, title="NRJMCMC-AIS-Arithmetic")

# ╔═╡ f0a86d89-07fd-4324-92d8-6b0e25c95bc6
traceplot(chain_nrjmcmc_ais_geom, title="NRJMCMC-AIS-Geometric")

# ╔═╡ 1a19ae1c-0fb1-4d34-b3f5-824f80109926
begin
	autocorplot(chain_rjmcmc_ais_arit,   label="RJMCMC-AIS-Arithmetic")
	autocorplot!(chain_rjmcmc_ais_geom,  label="RJMCMC-AIS-Geometric")
	autocorplot!(chain_nrjmcmc_ais_arit, label="NRJMCMC-AIS-Arithmetic")
	autocorplot!(chain_nrjmcmc_ais_geom, label="NRJMCMC-AIS-Geometric", legend=true)
end

# ╔═╡ 5a2d138f-1312-4b4c-b270-f87b3215abb2


# ╔═╡ Cell order:
# ╠═dfffe360-1907-4eb4-9e7d-0b911a61604f
# ╠═b2ead442-899f-11ee-07a6-433b9c11e97d
# ╠═32cf83fc-9a18-4899-b440-56395228dd8a
# ╠═e2a669a7-953a-4eeb-a410-6fe5dd092c50
# ╠═2f234dbd-f8e7-42b9-8c79-2fa01b017380
# ╠═f2fe90e0-5423-4c5b-b81d-1a5a77c6a0c4
# ╠═a0198327-8cb2-42e0-88a6-9a9e7e720e19
# ╠═66f95076-eaa9-4bac-bb93-4b11a8cde7bd
# ╠═258e721b-6d4a-45e0-9d4b-d5cf2271672b
# ╠═cbb941e2-379f-4f7e-9bdd-d738024bfed7
# ╠═4b5faec2-6570-4933-a2b4-4a986829468f
# ╠═e563fc61-282b-4252-933a-62d8232e23f5
# ╠═7453bd71-9e16-4c1b-8fc6-e5a02f93aed8
# ╠═e48e4872-9e22-4214-b213-26402e3ea19e
# ╠═45f35f82-e92c-4f1e-9b4f-908eabf3eae5
# ╠═f0a86d89-07fd-4324-92d8-6b0e25c95bc6
# ╠═1a19ae1c-0fb1-4d34-b3f5-824f80109926
# ╠═5a2d138f-1312-4b4c-b270-f87b3215abb2
