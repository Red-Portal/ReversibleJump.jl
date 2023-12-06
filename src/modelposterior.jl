
function estimate_conditional_acceptance_rate(stats::AbstractVector{<:NamedTuple})
    acc_rate = Dict{Pair{Int, Int}, OnlineStats.Mean}()
    for stat in stats
        k = stat.previous_order
        k′ = stat.proposal_order
        α = stat.jump_acceptance_rate

        key = (k => k′)
        if !haskey(acc_rate, key)
            acc_rate[key] = OnlineStats.Mean()
        end
        fit!(acc_rate[key], α)
    end
    acc_rate
end

function estimate_bayes_factors_raoblackwellized(
    acc_rate_estimates ::Dict{Pair{Int,Int},<:Any},
    order_counts       ::Dict{Int,Int},
    k_max              ::Int,
    max_log_bayesfactor::Real
)
    #=
        Estimate log Bayes factors:
            log BF_{0,1}, log BF_{1,2}, ..., log BF_{k_max,k_max+1},
        where
            BF_{k,k+1} = (∑_i^{N_k} α_i / N_k) / (∑_i^{N_{k+1}} α_i / N_{k+1})
    =##
    map(0:k_max) do k
        k′ = k+1
        k′_given_k_key = (k => k′)
        k_given_k′_key = (k′ => k)

        n_k = get(order_counts, k, 0)
        n_k′ = get(order_counts, k′, 0)

        ℓα_k_given_k′ = if haskey(acc_rate_estimates, k_given_k′_key)
            log(value(acc_rate_estimates[k_given_k′_key]))
        else
            -max_log_bayesfactor
        end

        ℓα_k′_given_k = if haskey(acc_rate_estimates, k′_given_k_key)
            log(value(acc_rate_estimates[k′_given_k_key]))
        else
            -max_log_bayesfactor
        end

        if n_k == 0 && n_k′ == 0
            ℓα_k_given_k′ = -max_log_bayesfactor
            ℓα_k′_given_k = -max_log_bayesfactor
        elseif n_k == 0
            ℓα_k_given_k′ = -max_log_bayesfactor
            ℓα_k′_given_k = 0.0
        elseif n_k′ == 0
            ℓα_k_given_k′ = 0.0
            ℓα_k′_given_k = -max_log_bayesfactor
        end
        ℓα_k_given_k′ - ℓα_k′_given_k
    end
end

function modelposterior(
    stats              ::AbstractVector{<:NamedTuple},
    orderprior         ::DiscreteDistribution,
    max_log_bayesfactor::Real = 100.0
)
    #=
        Bayes factor bridge-sampling estimator of Bartolucci et al. (2006, Eq.16)

        Model jumps are assumed to be continuous such as k → k+1 and k → k-1
    =##
    jump_stats    = filter(stat -> stat.move == :jump, stats)
    k_max         = maximum([stat.order for stat in jump_stats]) + 1

    order_posterior = [stat.order for stat in stats]
    order_counts    =
        [(i, count(==(i), order_posterior)) for i in unique(order_posterior)] |> Dict

    acc_rate_estimates    = estimate_conditional_acceptance_rate(jump_stats)
    log_bayes_factors_adj = estimate_bayes_factors_raoblackwellized(
        acc_rate_estimates, order_counts, k_max, max_log_bayesfactor
    )

    #=
        Bayes factor relative to the null model (k = 0 by convention):
            log BF_{0,0}, log BF_{0,1}, ..., log BF_{0,k_max+1},
        where
            log BF_{0,k} = log BF_{0,1} + log BF_{1,2} + ... + log BF_{k_max,k_max+1}.
    =##
    log_bayes_factor_00  = 0.0
    log_bayes_factors_0k = vcat(
        [log_bayes_factor_00], cumsum(log_bayes_factors_adj)
    )

    log_bayes_factors_k0 = -log_bayes_factors_0k

    #=
        Compute unnormalized posterior odds 
            PO = p(k|y) ∝ BF_{k,0} p(k) 
    =## 
    log_post_odds = log_bayes_factors_k0 + logpdf.(orderprior, 0:k_max+1)

    #=
        Compute Model probability 
            p(k|y) = BF_{k,0} / (1 + BF_{k,0} + BF_{k,1} + ... + BF_{k,k_max+1})
    =##
    log_norm         = logsumexp(log_post_odds)
    log_model_prob   = log_post_odds .- log_norm
    model_prob       = exp.(log_model_prob)

    sup = 0:k_max+1
    DiscreteNonParametric{
        eltype(sup),eltype(model_prob),typeof(sup),typeof(model_prob)
    }(sup, model_prob)
end

