
function sample(
    rng::Random.AbstractRNG,
    model, 
    θ_init,
    prior_order  ::Distributions.Distribution,
    kernel, 
    n_samples    ::Int;
    move_kernel         = (k, k′) -> 0.25*min(pdf(prior_order, k′)/pdf(prior_order, k), 1),
    show_progress::Bool = true,
    callback            = nothing
)
    #=
        Reversible Jump MCMC
        
        "use_ais" selects between AIS proposals and independent proposals

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible Jump MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.

        Peter J. Green
        "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination"
        Biometrika, 1995.
    =##

    θ       = θ_init
    θ_post  = Array{typeof(θ_init)}(undef, n_samples)
    ℓπ_hist = Array{Float64}(       undef, n_samples)
    prog    = if show_progress
        ProgressMeter.Progress(n_samples)
    else
        nothing
    end
    α_k     = OnlineStats.Mean()
    α_θ     = OnlineStats.Mean()

    n_target_calls = 0
    target_counted(θ) = begin
        n_target_calls += 1
        logdensity(model, θ)
    end

    for t = 1:n_samples
        if move == :birth || move == :death
            OnlineStats.fit!(α_k, stats.jump_rate)
        end
        OnlineStats.fit!(α_θ, stats.acceptance_rate)

        θ_post[t]  = deepcopy(θ)
        ℓπ_hist[t] = ℓπ

        if !isnothing(callback)
            callback(view(θ_post, 1:t))
        end

        if show_progress
            stats = (
                iter            = t,
                target_number   = model_order(model, θ),
                acceptance_rate = string(round(OnlineStats.value(α_θ), sigdigits=3)),
                jump_rate       = string(round(OnlineStats.value(α_k), sigdigits=3)),
                move            = move,
                n_target_calls  = n_target_calls,
                logjoint        = string(round(ℓπ,                     sigdigits=5)),
            )
            pm_next!(prog, stats)
        end
    end
    θ_post, ℓπ_hist
end

# function rjmcmc(
#     rng::Random.AbstractRNG,
#     model, 
#     θ_init,
#     prior_order  ::Distributions.Distribution,
#     kernel, 
#     n_samples    ::Int;
#     move_kernel         = (k, k′) -> 0.25*min(pdf(prior_order, k′)/pdf(prior_order, k), 1),
#     T            ::Int  = 10,
#     use_ais      ::Bool = true,
#     show_progress::Bool = true,
#     callback            = nothing
# )
#     rjmcmc(
#         rng, model, θ_init, kernel, n_samples;
#         move_kernel, T, use_ais, show_progress, callback
#     )
# end
