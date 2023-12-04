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
	using SignalAnalysis
	using Random
	using MCMCChains
	using DSP, WAV, SignalAnalysis
	using Statistics
	using MKL

	using Plots, StatsPlots
	plotly()
end


# ╔═╡ 6c491d4d-d0e1-4593-95c8-cc92466cf590
begin
    #=
	    Link: https://dosits.org/galleries/audio-gallery/anthropogenic-sounds/wind-turbines/
	    Underwater recording taken less than 50 m from a 500 kW turbine operating at the Bockstigen-Valar offshore wind farm (Baltic Sea, Sweden). Sounds were recorded at a water depth of 10 m.
	    Sound credit: Aarhus University, National Centre for Energy and Environment.      Sound file released under Creative Commons license, NonCommercial-NoDerivs.
	=##
	fname = "windturbine_sound.wav"
	y_raw, fs_raw = WAV.wavread(fname)
	@info(fname, size(y_raw), fs_raw)
	y_raw = y_raw[:,1]
	psd(y_raw; fs_raw, ylims=[-30, 0])
end

# ╔═╡ 6829b358-4647-4482-a4d6-5737e798db81
# ╠═╡ disabled = true
#=╠═╡
begin
	responsetype = Highpass(256; fs=fs_raw)
	designmethod = FIRWindow(DSP.Windows.blackman(513, zerophase=true))
	filter       = digitalfilter(responsetype, designmethod)
	y_hp = filt(filter, y_raw)
end
  ╠═╡ =#

# ╔═╡ f3b12345-3940-4a20-b1b8-db0f595bbab1
begin
	fs_target = 2000
		
	r  = fs_target/fs_raw
	y  = resample(y_raw, r)
	fs = fs_raw*r
	@info(size(y), fs)
	psd(y; fs, xlims=[0,fs/2])
end

# ╔═╡ 7c69aecb-2489-4cf0-9906-a1042535f5a5
begin
    y_sel  = y[10000:10064]
    y_sel /= std(y)
	N      = length(y_sel)
end

# ╔═╡ e2a669a7-953a-4eeb-a410-6fe5dd092c50
begin
	ν0    = 0.0
    γ0    = 0.0
    α_δ²  = 1/2
    β_δ²  = N/2
	α_λ   = 0.1
	β_λ   = 0.1 
	k_max = floor(Int, (N-1)/2)
    #prior = truncated(Geometric(0.2), upper=k_max)
	prior = truncated(NegativeBinomial(1/2 + α_λ, β_λ/(β_λ+1)), upper=k_max)
end

# ╔═╡ 3cdea5fa-a96d-4d39-af40-6b03bd4593c2
begin
    n_samples = 5000
	n_anneal  = 4

	#model  = SinusoidUnknownSNR(y_sel, ν0, γ0,  α_δ², β_δ², prior)
	#model  = SinusoidUnknownSNR(y_sel, ν0, γ0, α_δ², β_δ², prior)
	model  = SinusoidUnknownSNRReparam(y_sel, ν0, γ0, α_δ², β_δ², prior)

	path  = ArithmeticPath()
    prop  = SinusoidLocalProposal()
	#mcmc  = IMHRWMHSinusoid(model)
	#mcmc  = SliceSinusoid(SliceDoublingOut(), model, 0.2, 0.2)
    mcmc  = SliceSinusoid(SliceSteppingOut(), model, 0.5, 1.0)
	#mcmc  = SliceSinusoid(Slice(), model, 2.0, 2.0)

    #jump   = IndepJumpProposal(prop)
	jump   = AnnealedJumpProposal(n_anneal, prop, path)

	initial_params = [log(10.0)]
    initial_order  = 0
	
    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)
    #rjmcmc = ReversibleJump.ReversibleJumpMCMC(prior, jump, mcmc)

    samples, stats = ReversibleJump.sample(
        rjmcmc, model, n_samples, initial_order, initial_params; show_progress=false
    )
end

# ╔═╡ f1de0d2e-3082-494d-a73f-3124c6245143
begin
	Plots.plot([stat.order for stat in stats])
end

# ╔═╡ 3ad96bb2-886d-43a6-a4c2-066dfb425046
begin
	Plots.plot([sample[1] for sample in samples])
end

# ╔═╡ e4de6326-0a3e-44e3-86e5-3a6bb8a5f257
begin
	n_burn       = n_samples ÷ 2
	samples_burn = samples[n_burn+1:end]
	stats_burn   = stats[n_burn+1:end]
end

# ╔═╡ f17f33e8-db45-4a1d-94a7-bfe0556d9804
begin
	Plots.histogram([stat.order for stat in stats_burn], normed=true)
end

# ╔═╡ f303d31b-7646-47af-8c5b-94e78377114d
begin
	Plots.histogram([exp(sample[1]) for sample in samples_burn], bins=range(1,100; length=30))
end

# ╔═╡ 983e6e1e-86b5-4a22-b004-0f9ef26dd93a
f = [sample[2:end] for sample in samples_burn]

# ╔═╡ c76e555b-470c-47f2-8f1b-1a017395dc84
begin
	mixture, labeled = ReversibleJump.relabel(Random.default_rng(), f, 4; show_progress=false)
end

# ╔═╡ bfc9e429-8578-4875-9a10-fc3b4d8a5983
begin
	Plots.stephist(vcat(f...), bins=256, normed=true)
	Plots.plot!(range(0, π; length=4096), x -> pdf(mixture, x), xlims=[0,π])
end

# ╔═╡ 399c32f7-dbc9-4e38-bc36-d0914acc1078
begin
	Plots.plot(mixture, xlims=[0,π])
	Plots.histogram!(labeled[1], bins=range(0, π; length=256), normed=true, alpha=0.5)
	Plots.histogram!(labeled[2], bins=range(0, π; length=256), normed=true, alpha=0.5)
	Plots.histogram!(labeled[3], bins=range(0, π; length=256), normed=true, alpha=0.5)
	Plots.histogram!(labeled[4], bins=range(0, π; length=256), normed=true, alpha=0.5)
end

# ╔═╡ Cell order:
# ╠═dfffe360-1907-4eb4-9e7d-0b911a61604f
# ╠═b2ead442-899f-11ee-07a6-433b9c11e97d
# ╠═6c491d4d-d0e1-4593-95c8-cc92466cf590
# ╠═6829b358-4647-4482-a4d6-5737e798db81
# ╠═f3b12345-3940-4a20-b1b8-db0f595bbab1
# ╠═7c69aecb-2489-4cf0-9906-a1042535f5a5
# ╠═e2a669a7-953a-4eeb-a410-6fe5dd092c50
# ╠═3cdea5fa-a96d-4d39-af40-6b03bd4593c2
# ╠═f1de0d2e-3082-494d-a73f-3124c6245143
# ╠═3ad96bb2-886d-43a6-a4c2-066dfb425046
# ╠═e4de6326-0a3e-44e3-86e5-3a6bb8a5f257
# ╠═f17f33e8-db45-4a1d-94a7-bfe0556d9804
# ╠═f303d31b-7646-47af-8c5b-94e78377114d
# ╠═983e6e1e-86b5-4a22-b004-0f9ef26dd93a
# ╠═c76e555b-470c-47f2-8f1b-1a017395dc84
# ╠═bfc9e429-8578-4875-9a10-fc3b4d8a5983
# ╠═399c32f7-dbc9-4e38-bc36-d0914acc1078
