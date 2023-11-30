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


# ╔═╡ e2a669a7-953a-4eeb-a410-6fe5dd092c50
begin
	ν0    = 0.0
    γ0    = 0.0
    α_δ²  = 1/2
    β_δ²  = 128/2

    prior = Geometric(0.2)
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
    y_sel  = y[10000:10128]
    y_sel /= std(y)
	N      = length(y_sel)
end

# ╔═╡ 3cdea5fa-a96d-4d39-af40-6b03bd4593c2
begin
    n_samples = 2000
	n_anneal  = 8

	path  = ArithmeticPath()
    prop  = SinusoidUniformLocalProposal()
    #mcmc  = IMHRWMHUnknownSNR(y_sel, N)
	#mcmc  = IMHRWMHUnknownSNR(Uniform(0, π), N)
	mcmc  = SliceUnknownSNR(SliceSteppingOut(1.0), 1.0, 1.0)

    #jump   = IndepJumpProposal(prop)
	jump   = AnnealedJumpProposal(n_anneal, prop, path)

	initial_params = [log(10.0)]
    initial_order  = 0
	
    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.8)
    #rjmcmc = ReversibleJump.ReversibleJumpMCMC(prior, jump, mcmc)

	k_max  = floor(Int, (N-1)/2)
	#model  = SinusoidUnknownSNR(
	#	y_sel, ν0, γ0,  α_δ², β_δ², truncated(Geometric(0.2), upper=k_max) 
	#)
	# model  = SinusoidUnknownSNR(
	# 	y_sel, ν0, γ0, α_δ², β_δ², truncated(Geometric(0.5), upper=k_max) 
	# )
	model  = SinusoidUnknownSNRReparam(
		y_sel, ν0, γ0, α_δ², β_δ², truncated(Geometric(0.5), upper=k_max) 
	)
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

# ╔═╡ 5b209b3d-511a-4b7e-8fd5-1bae95d3f418
begin
    ω = vcat([sample[2:end] for sample in samples_burn]...)
	Plots.histogram(ω*fs/2/π, bins=256, normed=true)
end

# ╔═╡ Cell order:
# ╠═dfffe360-1907-4eb4-9e7d-0b911a61604f
# ╠═b2ead442-899f-11ee-07a6-433b9c11e97d
# ╠═e2a669a7-953a-4eeb-a410-6fe5dd092c50
# ╠═6c491d4d-d0e1-4593-95c8-cc92466cf590
# ╠═6829b358-4647-4482-a4d6-5737e798db81
# ╠═f3b12345-3940-4a20-b1b8-db0f595bbab1
# ╠═7c69aecb-2489-4cf0-9906-a1042535f5a5
# ╠═3cdea5fa-a96d-4d39-af40-6b03bd4593c2
# ╠═f1de0d2e-3082-494d-a73f-3124c6245143
# ╠═3ad96bb2-886d-43a6-a4c2-066dfb425046
# ╠═e4de6326-0a3e-44e3-86e5-3a6bb8a5f257
# ╠═f17f33e8-db45-4a1d-94a7-bfe0556d9804
# ╠═f303d31b-7646-47af-8c5b-94e78377114d
# ╠═5b209b3d-511a-4b7e-8fd5-1bae95d3f418
