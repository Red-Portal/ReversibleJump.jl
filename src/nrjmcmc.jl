
struct NonReversibleBirthDeath{
    Jump    <: AbstractJumpProposal
    Sampler <: AbstractMCMC.AbstractSampler
    UpRate  <: Real
}
    update_prob  ::UpRate
    jump_proposal::Jump
    inner_sampler::Sampler
end

function NonReversibleBirthDeathState
    move_direction::Int
    jump_state    ::JumpState
end

function AbstractMCMC.step(
    rng  ::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
         ::NonReversibleBirthDeath;
    initial_params,
    kwargs...,
)
    NonReversibleBirthDeathState(
        rand(rng, Bernoulli()),
        JumpState(
            initial_params, 
            logdensity(model, initial_params),
            model_order(model, initial_params),
            NamedTuple()
        )
    )
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.AbstractModel,
    sampler::NonReversibleBirthDeath,
    prev   ::NonReversibleBirthDeathState;
    kwargs...,
)
    k        = prev.state.order
    move_dir = prev.move_dir

    move = if rand(rng) ≤ τ
        :update
    elseif move_dir == true || k == 0
        :birth
    else
        :death
    end

    next = step_jump_move(
        rng, move, jump_proposal, inner_sampler, prev, model, move_kernel
    )

    if !next.stat.has_jumped
        move_dir = !move_dir
    end

    next.param, NonReversibleBirthDeathState(move_dir, next)
end

# function nrjmcmc(rng          ::Random.AbstractRNG,
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
