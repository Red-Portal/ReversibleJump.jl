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
	using MCMCChains
	using DelimitedFiles
	using Statistics
	using SignalAnalysis


	using Plots, StatsPlots
	plotly()
end


# ╔═╡ 6c491d4d-d0e1-4593-95c8-cc92466cf590
begin
    #=
	    Source: WDC-SILSO, Royal Observatory of Belgium, Brussels
	    link: https://www.sidc.be/SILSO/datafiles#total
	    accessed: 29/11/2023
	=##
	fname = "sunspot_monthly.csv"
	data  = readdlm(fname, ';', Float64)
	t_raw = data[:,3]
	y_raw = data[:,4]
	@info(fname, size(data))
	psd(y_raw)
end

# ╔═╡ 1a08a090-df94-40c6-b9b7-34ffb53a1aac
begin
	y_cent = y_raw .- mean(y_raw)
	T_max  = 200
	t, y   = t_raw[1:1:T_max], y_cent[1:1:T_max]
	N      = length(y)
end

# ╔═╡ d9282acb-a868-40dc-a4e8-e7b5c2a68d2a
plot(t, y)

# ╔═╡ e2a669a7-953a-4eeb-a410-6fe5dd092c50
begin
	ν0    = 0.0
    γ0    = 0.0
    α_δ²  = 1/2
    β_δ²  = N/2
	α_λ   = 0.1
	β_λ   = 0.1 
	n_anneal = 8
	k_max = floor(Int, (N-1)/2)
	prior = truncated(NegativeBinomial(1/2 + α_λ, β_λ/(β_λ+1)), upper=k_max)
    #prior = truncated(Geometric(0.8), upper=k_max)
	
	model  = SinusoidUnknownSNRReparam(y, ν0, γ0,  α_δ², β_δ², prior)
	#model  = SinusoidUnknownSNR(y, ν0, γ0,  α_δ², β_δ², prior)

    path  = ArithmeticPath()
    prop  = SinusoidLocalProposal()
    #mcmc  = IMHRWMHSinusoid(model)
	mcmc  = SliceSinusoid(SliceSteppingOut(), model, 0.5, 1.0)

    #jump   = IndepJumpProposal(prop)
	jump   = AnnealedJumpProposal(n_anneal, prop, path)
end

# ╔═╡ 3cdea5fa-a96d-4d39-af40-6b03bd4593c2
begin
    n_samples    = 3000

	initial_params = Float64[log(10.0)]
    initial_order  = 0
	
    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.8)
	#rjmcmc = ReversibleJump.ReversibleJumpMCMC(prior, jump, mcmc)

    samples, stats = ReversibleJump.sample(
        rjmcmc, model, n_samples, initial_order, initial_params; show_progress=false
    )
end

# ╔═╡ f1de0d2e-3082-494d-a73f-3124c6245143
begin
	Plots.plot([stat.order for stat in stats[1:1:end]])
end

# ╔═╡ 507498c4-67df-4d9b-b9d3-3efb6a1793eb
begin
	Plots.plot([stat.logtarget for stat in stats])
end

# ╔═╡ f17f33e8-db45-4a1d-94a7-bfe0556d9804
begin
	Plots.histogram([stat.order for stat in stats], normed=true)
end

# ╔═╡ 3ad96bb2-886d-43a6-a4c2-066dfb425046
begin
	Plots.plot([sample[1] for sample in samples[1:1:end]])
end

# ╔═╡ f303d31b-7646-47af-8c5b-94e78377114d
begin
	Plots.histogram([sample[1] for sample in samples])
end

# ╔═╡ 37475586-f64d-475a-b304-86ecb2da728f
begin
    ω = vcat([sample[2:end] for sample in samples]...)
	Plots.histogram(ω, bins=512, normed=true)
end

# ╔═╡ Cell order:
# ╠═dfffe360-1907-4eb4-9e7d-0b911a61604f
# ╠═b2ead442-899f-11ee-07a6-433b9c11e97d
# ╠═6c491d4d-d0e1-4593-95c8-cc92466cf590
# ╠═1a08a090-df94-40c6-b9b7-34ffb53a1aac
# ╠═d9282acb-a868-40dc-a4e8-e7b5c2a68d2a
# ╠═e2a669a7-953a-4eeb-a410-6fe5dd092c50
# ╠═3cdea5fa-a96d-4d39-af40-6b03bd4593c2
# ╠═f1de0d2e-3082-494d-a73f-3124c6245143
# ╠═507498c4-67df-4d9b-b9d3-3efb6a1793eb
# ╠═f17f33e8-db45-4a1d-94a7-bfe0556d9804
# ╠═3ad96bb2-886d-43a6-a4c2-066dfb425046
# ╠═f303d31b-7646-47af-8c5b-94e78377114d
# ╠═37475586-f64d-475a-b304-86ecb2da728f
