
abstract type AbstractSliceSampling end

struct SliceDoublingOut{F <: Real} <: AbstractSliceSampling
    max_doubling_out::Int
    window          ::F
end

SliceDoublingOut(window::Real) = SliceDoublingOut(8, window)

struct SliceSteppingOut{F <: Real} <: AbstractSliceSampling
    max_stepping_out::Int
    window          ::F
end

SliceSteppingOut(window::Real) = SliceSteppingOut(32, window)

struct Slice{F <: Real} <: AbstractSliceSampling
    window::F
end

function find_interval(
    rng::Random.AbstractRNG,
    alg::Slice,
       ::GibbsObjective,
       ::Real,
    θ₀ ::Real,
)
    w = alg.window
    u = rand(rng)
    L = θ₀ - w*u
    R = L + w
    L, R, 0
end

function find_interval(
    rng  ::Random.AbstractRNG,
    alg  ::SliceDoublingOut,
    model::GibbsObjective,
    ℓy   ::Real,
    θ₀   ::Real,
)
    #=
        Doubling out procedure for finding a slice
        (An acceptance rate < 1e-4 is treated as a potential infinite loop)

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    w = alg.window
    p = alg.max_doubling_out

    u = rand(rng)
    L = θ₀ - w*u
    R = L + w

    ℓπ_L = logdensity(model, L)
    ℓπ_R = logdensity(model, R)
    K    = 2

    for _ = 1:p
        if ((ℓy ≥ ℓπ_L) && (ℓy ≥ ℓπ_R))
            break
        end
        v = rand(rng)
        if v < 0.5
            L    = L - (R - L)
            ℓπ_L = logdensity(model, L)
        else
            R    = R + (R - L)
            ℓπ_R = logdensity(model, R)
        end
        K += 1
    end
    L, R, K
end

function find_interval(
    rng  ::Random.AbstractRNG,
    alg  ::SliceSteppingOut,
    model::GibbsObjective,
    ℓy   ::Real,
    θ₀   ::Real,
    )
    #=
        Stepping out procedure for finding a slice
        (An acceptance rate < 1e-4 is treated as a potential infinite loop)

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    w = alg.window
    m = alg.max_stepping_out

    u      = rand(rng)
    L      = θ₀ - w*u
    R      = L + w
    V      = rand(rng)
    J      = floor(m*V)
    K      = (m - 1) - J 
    n_eval = 0

    while J > 0 && ℓy < logdensity(model, L)
        L = L - w
        J = J - 1
        n_eval += 1
    end
    while K > 0 && ℓy < logdensity(model, R)
        R = R + w
        K = K - 1
        n_eval += 1
    end
    L, R, n_eval
end

accept_slice_proposal(
    ::AbstractSliceSampling,
    ::GibbsObjective,
    ::Real,
    ::Real,
    ::Real,
    ::Real,
    ::Real,
) = true

function accept_slice_proposal(
    alg  ::SliceDoublingOut,
    model::GibbsObjective,
    ℓy   ::Real,
    θ₀   ::Real,
    θ₁   ::Real,
    L    ::Real,
    R    ::Real,
) 
    #=
        acceptance rule for the doubling procedure

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    w    = alg.window
    D    = false
    ℓπ_L = logdensity(model, L)
    ℓπ_R = logdensity(model, R)

    while R - L > 1.1*w
        M = (L + R)/2
        if (θ₀ < M && θ₁ ≥ M) || (θ₀ ≥ M && θ₁ < M)
            D = true
        end

        if θ₁ < M
            R    = M
            ℓπ_R = logdensity(model, R)
        else
            L    = M
            ℓπ_L = logdensity(model, L)
        end

        if D && ℓy ≥ ℓπ_L && ℓy ≥ ℓπ_R
            return false
        end
    end
    true
end

function slice_sampling_univariate(
    rng  ::Random.AbstractRNG,
    alg  ::AbstractSliceSampling,
    model::GibbsObjective, 
    θ₀   ::Real,
    ℓπ₀  ::Real)
    #=
        Univariate slice sampling kernel
        (An acceptance rate < 1e-4 is treated as a potential infinite loop)

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    u  = rand(rng)
    ℓy = log(u) + ℓπ₀

    L, R, n_prop = find_interval(rng, alg, model, ℓy, θ₀)

    while true
        U      = rand(rng)
        θ′      = L + U*(R - L)
        ℓπ′     = logdensity(model, θ′)
        n_prop += 1
        if (ℓy < ℓπ′) && accept_slice_proposal(alg, model, ℓy, θ₀, θ′, L, R)
            return θ′, ℓπ′, 1/n_prop
        end

        if θ′ < θ₀
            L = θ′
        else
            R = θ′
        end

        if n_prop > 10000 
            @error("Too many rejections. Something looks broken. \n θ = $(θ₀) \n ℓπ = $(ℓπ₀)")
        end
    end
end

function slice_sampling(
    rng      ::Random.AbstractRNG,
    alg      ::AbstractSliceSampling,
    model, 
    θ        ::AbstractVector,
)
    ℓp    = logdensity(model, θ)
    ∑acc  = 0.0
    n_acc = 0
    k     = length(θ)
    idxs  = randperm(rng, k) # the kernel is not reversible without random permutation
    for idx in idxs
        model_gibbs = GibbsObjective(model, idx, θ)
        θ′idx, ℓp, acc = slice_sampling_univariate(rng, alg, model_gibbs, θ[idx], ℓp)
        ∑acc  += acc
        n_acc += 1
        θ[idx] = θ′idx
    end
    avg_acc = n_acc > 0 ? ∑acc/n_acc : 1
    θ, ℓp, avg_acc
end

# function kernel_slice_gibbs(target::Function, 
#                             θ     ::DoAParamType,
#                             window::NamedTuple,
#                             supp  ::NamedTuple;
#                             prng  ::Random.AbstractRNG = Random.GLOBAL_PRNG)
#     #= 
#         Slice-in-Gibbs 

#         Update all parameters one at a time
#     =##

#     ∑acc  = 0.0
#     n_acc = 0
#     ℓp    = target(θ)::Real

#     lv1_fieldnames = fieldnames(typeof(θ))
#     lv1_lenses     = map(PropertyLens, lv1_fieldnames)

#     for (lv1_name, lv1_lens) ∈ zip(lv1_fieldnames, lv1_lenses)
#         lv2_fieldnames = fieldnames(typeof(lv1_lens(θ)))
#         lv2_lenses     = map(PropertyLens, lv2_fieldnames)

#         for lv2_lens ∈ lv2_lenses
#             param_lens = opcompose(lv1_lens, lv2_lens)

#             θ₀_spec   = param_lens(θ)
#             wind_spec = param_lens(window)
#             supp_spec = param_lens(supp)

#             target_spec(θ′) = target(set(θ, param_lens, θ′))

#             θ′, ℓp, acc = slice_sampling(
#                 target_spec, 
#                 θ₀_spec,
#                 ℓp, 
#                 wind_spec, 
#                 supp_spec;
#                 prng)

#             ∑acc  += acc
#             n_acc += 1

#             θ = set(θ, param_lens, θ′)
#         end
#     end
#     θ, ℓp, (acceptance_rate = n_acc > 0 ? ∑acc/n_acc : 1,)
# end

# function kernel_imh_rwmh(target, 
#                          θ::DoAParamType;
#                          q_local_indep,
#                          n_total_acc_param=nothing,
#                          λ_mix = 0.2,
#                          σ_rw  = 0.1,
#                          prng=Random.GLOBAL_PRNG)
#     #= 
#       independent Metropolis-Hastings & Random Walk Metropolis Hastings mixture kernel

#       Only updates the local parameters.
      
#     =##

#     # for key ∈ keys(θ.globals)
#     #     θ, ℓp, _ = kernel_slice_local_univariate(
#     #     θ, target, ℓp, key, idx, window; prng=prng)
#     # end

#     K           = length(θ.locals[1])
#     local_keys  = keys(θ.locals)
#     supports    = [key => support(q_local_indep[key]) for key ∈ local_keys]
#     supp_ub     = NamedTuple(key => Float64(maximum(sup)) for (key, sup) ∈ supports)
#     supp_lb     = NamedTuple(key => Float64(minimum(sup)) for (key, sup) ∈ supports)
#     ℓπ          = target(θ)
#     θ′          = deepcopy(θ)
#     n_acc       = 0
#     n_acc_param = Dict(zip(local_keys, zeros(Int64, length(local_keys))))

#     if σ_rw isa Real
#         σ_rw = NamedTuple(zip(local_keys, fill(σ, lenght(local_keys))))
#     end

#     for k = 1:K 
#         for key ∈ local_keys
#             θₖ = θ.locals[key][k]
#             θₖ′, ℓq′, ℓq = if rand(prng, Bernoulli(λ_mix))
#                 # Independent proposal
#                 θₖ′ = rand(prng, q_local_indep[key])
#                 θₖ′, logpdf(q_local_indep[key], θₖ′), logpdf(q_local_indep[key], θₖ)
#             else
#                 # Dependent proposal
#                 qₖ  = truncated(Normal(θₖ, σ_rw[key]), supp_lb[key], supp_ub[key])
#                 θₖ′ = rand(prng, qₖ)
#                 θₖ′, logpdf(qₖ, θₖ′), logpdf(qₖ, θₖ)
#             end

#             θ′.locals[key][k] = θₖ′

#             ℓπ′ = target(θ′)

#             ℓα = (ℓπ′ - ℓπ) - (ℓq′ - ℓq)
#             if ℓα > log(rand(prng))
#                 θ      = θ′
#                 ℓπ     = ℓπ′
#                 n_acc += 1
#                 n_acc_param[key] += 1
#             else
#                 θ′.locals[key][k] = θₖ
#             end
#         end
#     end

#     if !isnothing(n_total_acc_param)
#         for key ∈ local_keys
#            n_total_acc_param[key] += n_acc_param[key]
#         end
#     end
#     θ, ℓπ, (acceptance_rate = n_acc/K,)
# end

# function mcmc(target::Function, 
#               θ_init::DoAParamType,
#               kernel::Function, 
#               n_samples::Int;
#               prng      = Random.GLOBAL_PRNG,
#               callback! = nothing)
#     θ       = θ_init
#     θ_post  = Array{typeof(θ_init)}(undef, n_samples)
#     ℓπ_hist = Array{Float64}(       undef, n_samples)
#     prog    = ProgressMeter.Progress(n_samples)
#     α_avg   = OnlineStats.Mean()
#     for t = 1:n_samples
#         θ, ℓπ, stats = kernel(target, θ; prng=prng)

#         OnlineStats.fit!(α_avg, stats.acceptance_rate)

#         # display(θ_post[1:t])

#         # if t > 5
#         #     throw()
#         # end

#         θ_post[t]  = θ
#         ℓπ_hist[t] = ℓπ

#         if !isnothing(callback!)
#             callback!(view(θ_post, 1:t))
#         end

#         ProgressMeter.next!(
#             prog;
#             showvalues = [(:iter,            t), 
#                           (:acceptance_rate, string(round(OnlineStats.value(α_avg), sigdigits=3))), 
#                           (:logjoint,        string(round(ℓπ,                       sigdigits=5))),
#                           ])
#     end
#     θ_post, ℓπ_hist
# end
