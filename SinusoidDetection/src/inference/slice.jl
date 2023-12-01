
abstract type AbstractSliceSampling <: AbstractMCMC.AbstractSampler end

struct SliceDoublingOut{W <: AbstractVector} <: AbstractSliceSampling
    max_doubling_out::Int
    window          ::W
end

SliceDoublingOut(window::AbstractVector) = SliceDoublingOut(8, window)

struct SliceSteppingOut{W <: AbstractVector} <: AbstractSliceSampling
    max_stepping_out::Int
    window          ::W
end

SliceSteppingOut(window::AbstractVector) = SliceSteppingOut(32, window)

struct Slice{W <: AbstractVector} <: AbstractSliceSampling
    window::W
end

function find_interval(
    rng::Random.AbstractRNG,
       ::Slice,
       ::GibbsObjective,
    w  ::Real,
       ::Real,
    θ₀ ::Real,
)
    u = rand(rng)
    L = θ₀ - w*u
    R = L + w
    L, R, 0
end

function find_interval(
    rng  ::Random.AbstractRNG,
    alg  ::SliceDoublingOut,
    model::GibbsObjective,
    w    ::Real,
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
    w    ::Real,
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
    ::Real,
) = true

function accept_slice_proposal(
         ::SliceDoublingOut,
    model::GibbsObjective,
    w    ::Real,
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
    w    ::Real,
    θ₀   ::Real,
    ℓπ₀  ::Real
)
    #=
        Univariate slice sampling kernel
        (An acceptance rate < 1e-4 is treated as a potential infinite loop)

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    u  = rand(rng)
    ℓy = log(u) + ℓπ₀

    L, R, n_prop = find_interval(rng, alg, model, w, ℓy, θ₀)

    while true
        U      = rand(rng)
        θ′      = L + U*(R - L)
        ℓπ′     = logdensity(model, θ′)
        n_prop += 1
        if (ℓy < ℓπ′) && accept_slice_proposal(alg, model, w, ℓy, θ₀, θ′, L, R)
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
    w = alg.window
    @assert length(w) == length(θ)
    
    ℓp    = logdensity(model, θ)
    ∑acc  = 0.0
    n_acc = 0
    for idx in 1:length(θ)
        model_gibbs = GibbsObjective(model, idx, θ)
        θ′idx, ℓp, acc = slice_sampling_univariate(
            rng, alg, model_gibbs, w[idx], θ[idx], ℓp
        )
        ∑acc  += acc
        n_acc += 1
        θ[idx] = θ′idx
    end
    avg_acc = n_acc > 0 ? ∑acc/n_acc : 1
    θ, ℓp, avg_acc
end

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
