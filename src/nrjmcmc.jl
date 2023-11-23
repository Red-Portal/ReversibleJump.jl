
struct NonReversibleJumpMCMC{
    JumpProp   <: AbstractJumpProposal,
    MovePairs  <: Vector{<:AbstractJumpMovePair},
    MoveWeight <: StatsBase.AbstractWeights,
    UpRate     <: Real, 
    MCMCKern,
}
    jump_proposal::JumpProp
    move_pairs   ::MovePairs
    move_weights ::MoveWeight
    mcmc_kernel  ::MCMCKern
    update_rate  ::UpRate
end

function NonReversibleJumpMCMC(
    jump_proposal::AbstractJumpProposal,
    mcmc_kernel,
    move_pairs   ::AbstractVector{<:AbstractJumpMovePair} = [BirthDeath()],
    move_weights ::StatsBase.AbstractWeights              = pweights(fill(1.0, length(move_pairs))),
    jump_rate    ::Real                                   = 0.5, 
)
    @assert length(move_weights) == length(move_pairs)
    NonReversibleJumpMCMC(
        jump_proposal, move_pairs, move_weights, mcmc_kernel, 1 - jump_rate
    )
end

function AbstractMCMC.step(
    rng  ::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
         ::NonReversibleJumpMCMC;
    initial_params,
    initial_order,
    kwargs...,
)
    initial_params, NRJState(
        rand(rng) > 0.5,
        initial_params, 
        logdensity(model, initial_params),
        initial_order,
        NamedTuple()
    )
    
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.AbstractModel,
    sampler::NonReversibleJumpMCMC,
    prev   ::NRJState;
    kwargs...,
)
    @unpack jump_proposal, move_pairs, move_weights, mcmc_kernel, update_rate = sampler

    move_pair   = sample(rng, move_pairs, move_weights)
    k           = prev.order
    move_fwd, _ = forward_move(move_pair, k)
    move_bwd, _ = backward_move(move_pair, k)
    direction   = prev.direction

    next = if rand(rng) ≤ update_rate
        next_param, lp = transition_mcmc(rng, mcmc_kernel, model, prev.param)
        setproperties(prev, (param = next_param,
                             lp    = lp,
                             stats = (move = :update,)))
    else
        move = if direction || k == 0
            move_fwd
        else
            move_bwd
        end

        next = transition_jump(
            rng, move, jump_proposal, prev, mcmc_kernel, model, (k, k′) -> 1.0
        )       
        if !next.stats.jump_accepted
            @set next.direction = !direction
        else
            next
        end
    end
    next.param, next
end

# function sampler(rng          ::Random.AbstractRNG,
#                  model,
#                  θ_init,
#                  τ            ::Real,
#                  kernel, 
#                  n_samples    ::Int;
#                  show_progress::Bool = true,
#                  T            ::Int  = 10,
#                  use_ais      ::Bool = true,
#                  callback            = nothing)
#     #=
#         Non-Reversible Jump MCMC

#         P. Gagnon, A. Doucet,
#         "Nonreversible Jump Algorithms for Bayesian Nested Model Selection,"
#         Journal of Computational and Graphical Statistics, 2021.
#     =##
#     θ       = θ_init
#     θ_post  = Array{typeof(θ_init)}(undef, n_samples)
#     ℓπ_hist = Array{Float64}(       undef, n_samples)
#     prog    = if show_progress
#         ProgressMeter.Progress(n_samples)
#     else
#         nothing
#     end
#     α_k     = OnlineStats.Mean()
#     α_θ     = OnlineStats.Mean()

#     n_target_calls = 0
#     target_counted(θ) = begin
#         n_target_calls += 1
#         target(θ)
#     end

#     move_kernel   = (k, k′) -> 1.0
#     move_dir      = rand(rng, Bernoulli()) ? +1 : -1
#     ℓπ            = logdensity(model, θ_init)
#     for t = 1:n_samples
#         k = model_order(model, θ)

#         move = if rand(rng) ≤ τ
#             :update
#         elseif move_dir == 1 || k == 0
#             :birth
#         else
#             :death
#         end

#         θ, ℓπ, stats = kernel_birthdeath(
#             rng, move, θ, target_counted, move_kernel, kernel, use_ais, T)

#         if move == :update
#             OnlineStats.fit!(α_θ, stats.acceptance_rate)
#         else
#             if !stats.has_jumped
#                 move_dir *= -1
#             end
#             OnlineStats.fit!(α_θ, stats.acceptance_rate)
#             OnlineStats.fit!(α_k, stats.jump_rate)
#         end

#         θ_post[t]  = deepcopy(θ)
#         ℓπ_hist[t] = ℓπ

#         if !isnothing(callback)
#             callback(view(θ_post, 1:t))
#         end

#         if show_progress
#             stats = (
#                 iter            = t,
#                 target_number   = model_order(model, θ),
#                 acceptance_rate = string(round(OnlineStats.value(α_θ), sigdigits=3)),
#                 jump_rate       = string(round(OnlineStats.value(α_k), sigdigits=3)),
#                 n_target_calls  = n_target_calls,
#                 move            = move,
#                 logjoint        = string(round(ℓπ,                     sigdigits=5)),
#             )
#             pm_next!(prog, stats)
#         end
#     end
#     θ_post, ℓπ_hist
# end

# function nrjmcmc(
#     model,
#     θ_init,
#     τ            ::Real,
#     kernel, 
#     n_samples    ::Int;
#     show_progress::Bool = true,
#     T            ::Int  = 10,
#     use_ais      ::Bool = true,
#     callback            = nothing
# ) 
#     nrjmcmc(
#         Random.default_rng(), model, θ_init, τ, kernel, n_samples;
#         show_progress, T, use_ais, callback
#     )
# end
