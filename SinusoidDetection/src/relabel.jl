
function unshuffle(
    z           ::AbstractVector{<:Real},
    shuffle_idxs::AbstractVector{<:Integer}
)
    z_unshuffle = zeros(Int, length(z))
    for (idx, shuffle_idx) in enumerate(shuffle_idxs)
        z_unshuffle[shuffle_idx] = z[idx]
    end
    z_unshuffle
end

function sem_propose_allocation(
    rng::AbstractRNG,
    x  ::AbstractVector{T},
    μ  ::AbstractVector{T},
    σ  ::AbstractVector{T},
    w  ::AbstractVector{T},
    λ  ::T
) where {T <: Real}
    L  = length(μ)
    ℓw = log.(w)

    @. ℓw[ℓw == zero(T)] = -eps(T)

    Θ   = π
    z   = zeros(Int, length(x))
    pc  = collect(1:L+1)
    ∑ℓg = 0
    ∑ℓq = 0

    shuffle_idxs = Random.randperm(rng, length(x))
    x_shuffle    = x[shuffle_idxs]

    for (j, xj) in enumerate(x_shuffle)
        ℓg = map(pc) do l
            if l == L+1
                log(λ) - log(Θ)
            else
                logpdf(Normal(μ[l], σ[l]), xj) + ℓw[l] - log1mexp(ℓw[l])
            end
        end
        ℓZ      = logsumexp(ℓg)
        ℓg_norm = @. exp(ℓg - ℓZ)
        pc_idx  = rand(rng, Categorical(ℓg_norm))
        zj      = pc[pc_idx]
        ∑ℓg    += ℓg_norm[pc_idx]
        ∑ℓq    += ℓg[pc_idx]
        z[j]    = zj
        if zj ≤ L
            deleteat!(pc, pc_idx)
        end
    end
    unshuffle(z, shuffle_idxs), ∑ℓg, ∑ℓq
end

function sem_sample_allocation(
    rng   ::Random.AbstractRNG,
    x     ::AbstractVector{T},
    μ     ::AbstractVector{T},
    σ     ::AbstractVector{T},
    w     ::AbstractVector{T},
    λ     ::T,
    n_iter::Int
) where {T <: Real}
    z, ∑ℓg, ∑ℓq = sem_propose_allocation(rng, x, μ, σ, w, λ)
    for _ in 1:n_iter
        z′, ∑ℓg′, ∑ℓq′ = sem_propose_allocation(rng, x, μ, σ, w, λ)
        ℓα = min((∑ℓq′ - ∑ℓg′) - (∑ℓq - ∑ℓg), 0)
        ℓu = -Random.randexp(rng, T)
        if ℓu < ℓα
            z   = z′
            ∑ℓg = ∑ℓg′
            ∑ℓq = ∑ℓq′
        end
    end
    z
end

function sem_initialize_mixtures(
    xs::AbstractVector{<:AbstractVector{T}}, L::Int
) where {T <: Real}
    xs        = map(sort, xs)
    M         = length(xs)
    M_l       = zeros(Int, L)
    x_relabel = zeros(T, L, M)
    for x in xs
        for (l, xj) ∈ enumerate(x)
            if l <= L
                M_l[l] += 1
                x_relabel[l,M_l[l]] = xj
            end
        end
    end
    μ = map(1:L) do l
        median(x_relabel[l, 1:M_l[l]])
    end
    σ = map(1:L) do l
        x_relabel_l = x_relabel[l, 1:M_l[l]]
        Q1, Q3      = quantile(x_relabel_l, [T(0.25), T(0.75)])
        (Q3 - Q1) / (2 * norminvcdf(T(0.75)))
    end
    μ, σ
end

function sem_fit(
    rng          ::Random.AbstractRNG,
    xs           ::AbstractVector{<:AbstractVector{T}},
    n_mixtures   ::Int,
    n_iter       ::Int,
    n_imh_iter   ::Int,
    show_progress::Bool
) where {T <: Real}
    M    = length(xs)
    L    = n_mixtures
    λ    = T(0.1)
    w    = fill(T(0.9)/L, L)
    prog = ProgressMeter.Progress(n_iter; enabled=show_progress)
    μ, σ = sem_initialize_mixtures(xs, L)

    for _ = 1:n_iter
        x_aligned = zeros(T, L, M)
        M_l       = zeros(Int, L)
        M_λ       = zero(T)
        for i = 1:M
            z = sem_sample_allocation(rng, xs[i], μ, σ, w, λ, n_imh_iter)
            for (j, zj) ∈ enumerate(z)
                if zj <= L
                    M_l[zj] += 1
                    x_aligned[zj, M_l[zj]] = xs[i][j]
                else
                    M_λ += 1
                end
            end
        end
        w = M_l / M
        λ = M_λ / M
        μ = map(1:L) do l
            if M_l[l] > 2
                median(x_aligned[l,1:M_l[l]])
            else
                T(0)
            end
        end
        σ = map(1:L) do l
            if M_l[l] > 2
                xsₗ     = x_aligned[l,1:M_l[l]]
                Q1, Q3 = quantile(xsₗ, [T(0.25), T(0.75)])
                (Q3 - Q1) / (2 * norminvcdf(T(0.75)))
            else
                T(10)
            end
        end
        next!(prog, showvalues=[
            (:state,:fit_mixture), (:μ,μ), (:σ,σ), (:w,w), (:λ,λ)
        ])
    end
    μ, σ, w, λ
end

function relabel(
    rng          ::Random.AbstractRNG,
    samples      ::AbstractVector{<:AbstractVector{T}},
    n_mixtures   ::Int;
    n_iter       ::Int  = 16,
    n_imh_iter   ::Int  = 32,
    show_progress::Bool = true
) where {T <: Real}
    μ, σ, w, λ = sem_fit(rng, samples, n_mixtures, n_iter, n_imh_iter, show_progress)

    labeled = [Array{T}(undef, 0) for _ in 1:n_mixtures]
    prog    = ProgressMeter.Progress(n_iter; enabled=show_progress)
    for (idx, sample) in enumerate(samples)
        labels = sem_sample_allocation(rng, sample, μ, σ, w, λ, n_imh_iter)

        for (j, xj) in enumerate(sample)
            label = labels[j]
            if label ≤ n_mixtures
                push!(labeled[label], xj)
            end
        end
        next!(prog, showvalues=[(:state,:relabel), (:sample_idx,idx)])
    end
    MixtureModel(Normal.(μ, σ), w/sum(w)), labeled
end
