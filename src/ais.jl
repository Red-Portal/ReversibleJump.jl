
abstract type AbstractAnnealingPath end

struct GeometricPath <: AbstractAnnealingPath end

struct ArithmeticPath <: AbstractAnnealingPath end

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

function anneal(::GeometricPath, ℓρ_T, ℓρ_0, t, T)
    γ = (t/T)
    γ*ℓρ_T + (1 - γ)*ℓρ_0
end

function anneal(::ArithmeticPath, ℓρ_T, ℓρ_0, t, T)
    ℓγ  = log(t/T)
    logaddexp(ℓρ_0 + log1mexp(ℓγ), ℓρ_T + ℓγ)
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

