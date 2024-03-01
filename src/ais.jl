
abstract type AbstractAnnealingPath end

# Brekelmans, Rob, et al. "Annealed importance sampling with q-paths." NeurIPS'20.
struct QPath{Q <: Real} <: AbstractAnnealingPath
    q::Q

    function QPath(q::Real)
        @assert 0 ≤ q ≤ 1
        new{typeof(q)}(q)
    end
end

function anneal(path::QPath, ℓρ_T, ℓρ_0, t, T)
    # ((1 - β)*ℓρ_0^(1 - q) + β*ℓρ_T^(1 - q))^(1/(1 - q))
    #
    # (1/(1 - q))*log(
    #     exp( log(1 - β) + (1 - q)*log(ℓρ_0) ) + exp(log(β) + (1 - q)*log(ℓρ_T) )
    # )

    q  = path.q
    β  = t/T
    ℓβ = log(β)

    if q == 1
        β*ℓρ_T + (1 - β)*ℓρ_0
    else
        1/(1 - q)*logaddexp(
            (1 - q)*ℓρ_0 + log1mexp(ℓβ), (1 - q)*ℓρ_T + ℓβ
        )
    end
end

GeometricPath()  = QPath(1.0)

ArithmeticPath() = QPath(0.0)

struct AnnealedJumpProposal{
    Prop,
    AnnealPath <: AbstractAnnealingPath
} <: AbstractJumpProposal
    n_annealed    ::Int
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
    T          ::Int
    inverse_map::INV
    fwd_density::FWD
    bwd_density::BWD
    path       ::AnnealPath
end

function logdensity(annealed::AnnealedTarget, θ)
    @unpack model, t, T, inverse_map, fwd_density, bwd_density, path = annealed
    ℓρ_T = logdensity(model, θ) + bwd_density(θ)
    ℓρ_0 = logdensity(model, inverse_map(θ)) + fwd_density(θ)
    anneal(path, ℓρ_T, ℓρ_0, t, T)
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
    @unpack n_annealed, path = jump 

    ℓr  = -ℓπ - ϕ_ktok′(θ)
    T   = n_annealed 

    target = AnnealedTarget(model, 0, T, G⁻¹, ϕ_ktok′, ϕ_k′tok, path)
    for t = 1:T-1
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

