
abstract type AbstractAnnealingPath end

struct QPath{Q <: Real} <: AbstractAnnealingPath
    q      ::Q
    n_steps::Int

    function QPath(q::Real, n_steps::Int)
        @assert 0 ≤ q ≤ 1
        new{typeof(q)}(q, n_steps)
    end
end

Base.length(path::QPath) = path.n_steps

function anneal(path::QPath, ℓρ_0, ℓρ_T, t)
    # ((1 - β)*ℓρ_0^(1 - q) + β*ℓρ_T^(1 - q))^(1/(1 - q))
    #
    # (1/(1 - q))*log(
    #     exp( log(1 - β) + (1 - q)*log(ℓρ_0) ) + exp(log(β) + (1 - q)*log(ℓρ_T) )
    # )

    T  = path.n_steps
    q  = path.q
    β  = t/T

    if q == 1
        β*ℓρ_T + (1 - β)*ℓρ_0
    else
        ℓβ = log(β)
        1/(1 - q)*logaddexp(
            (1 - q)*ℓρ_0 + log1mexp(ℓβ), (1 - q)*ℓρ_T + ℓβ
        )
    end
end

GeometricPath(n_steps::Int)  = QPath(1.0, n_steps)

ArithmeticPath(n_steps::Int) = QPath(0.0, n_steps)

struct CustomPath{S <: AbstractVector{<:Real}} <: AbstractAnnealingPath
    schedule::S

    function CustomPath(schedule::AbstractVector{<:Real})
        @assert all(@. 0 ≤ schedule ≤ 1)
        @assert first(schedule) == 0
        @assert last(schedule)  == 1
        new{typeof(schedule)}(schedule)
    end
end

Base.length(path::CustomPath) = length(path.schedule)

function anneal(path::CustomPath, ℓρ_0, ℓρ_T, t)
    γ = path.schedule[t]
    γ*ℓρ_T + (1 - γ)*ℓρ_0
end

struct AnnealedJumpProposal{
    Prop,
    AnnealPath <: AbstractAnnealingPath
} <: AbstractJumpProposal
    local_proposal::Prop
    path          ::AnnealPath
end

struct AnnealedTarget{
    Model,
    INV,
    FWD,
    BWD,
    AnnealPath <: AbstractAnnealingPath
}
    model      ::Model
    t          ::Int
    inverse_map::INV
    fwd_density::FWD
    bwd_density::BWD
    path       ::AnnealPath
end

function logdensity(annealed::AnnealedTarget, θ)
    @unpack model, t, inverse_map, fwd_density, bwd_density, path = annealed
    ℓρ_T = logdensity(model, θ) + bwd_density(θ)
    ℓρ_0 = logdensity(model, inverse_map(θ)) + fwd_density(θ)
    anneal(path, ℓρ_0, ℓρ_T, t)
end

function step_ais(
    rng   ::Random.AbstractRNG,
    jump  ::AnnealedJumpProposal,
    mcmc,
    model,
    θ,
    ℓπ,
    G⁻¹,
    ϕ_ktok′,
    ϕ_k′tok,
)
    #=
        Generic Annealed Importance Sampling Jump Proposal

        G. Karagiannis, C. Andrieu, 
        "Annealed Importance Sampling Reversible Jump MCMC Algorithms"
        in Journal of Computational and Graphical Statistics, 2013.
    =##
    @unpack path = jump 

    ℓr  = -ℓπ - ϕ_ktok′(θ)

    target = AnnealedTarget(model, 0, G⁻¹, ϕ_ktok′, ϕ_k′tok, path)
    for t in 1:length(path)-1
        target_annealed = @set target.t = t
        ℓρₜ = logdensity(target_annealed, θ)
        if !isfinite(ℓρₜ)
            break
        end
        θ, ℓρₜ′ = transition_mcmc(rng, mcmc, target_annealed, θ)
        ℓr    += ℓρₜ - ℓρₜ′
    end
    ℓπ′  = logdensity(model, θ)
    ℓr += ℓπ′ + ϕ_k′tok(θ)
    θ, ℓπ′, ℓr, NamedTuple()
end

