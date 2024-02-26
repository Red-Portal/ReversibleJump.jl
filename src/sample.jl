
function sample(
    rng           ::Random.AbstractRNG,
    sampler       ::AbstractRJMCMCSampler,
    model,
    n_samples     ::Int,
    initial_order ::Int,
    initial_params; 
    show_progress ::Bool = true,
    callback             = nothing
)
    param_chain  = Array{typeof(initial_params)}(undef, n_samples)
    stats_chain  = Array{NamedTuple}(            undef, n_samples)
    prog         = ProgressMeter.Progress(n_samples; enabled=show_progress, showspeed=true)
    avg_jump_acc = OnlineStats.Mean()

    _, state = AbstractMCMC.step(rng, model, sampler; initial_params, initial_order)
    for t = 1:n_samples
        param, state = AbstractMCMC.step(rng, model, sampler, state)
        stats = merge((iteration = t,
                       order     = state.order,
                       logtarget = state.lp),
                      state.stats)

        stats = if stats.move != :update
            fit!(avg_jump_acc, stats.jump_acceptance_rate)
            merge(stats, (average_jump_rate=value(avg_jump_acc),))
        else
            stats
        end

        if !isnothing(callback)
            stats′ = callback(param, stats)
            stats = merge(stats, stats′)
        end

        if show_progress
            pm_next!(prog, stats)
        end
        param_chain[t] = param
        stats_chain[t] = stats
    end
    param_chain, stats_chain
end


function sample(
    sampler       ::AbstractRJMCMCSampler,
    model,
    n_samples     ::Int,
    initial_order ::Int,
    initial_params; 
    show_progress ::Bool = true,
    callback             = nothing
)
    sample(
        Random.default_rng(),
        sampler,
        model,
        n_samples,
        initial_order,
        initial_params;
        show_progress,
        callback
    )
end
