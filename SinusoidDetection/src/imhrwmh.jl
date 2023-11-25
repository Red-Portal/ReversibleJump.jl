
function ReversibleJump.logdensity(obj::GibbsObjective, θi)
    @unpack model, idx, θ = obj
    θ′ = @set θ[idx] = θi
    logdensity(model, θ′)
end

function transition_imh(rng::Random.AbstractRNG, model, q, θ)
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    ℓw′ = ℓπ′ - logpdf(q, θ′)
    ℓw = ℓπ - logpdf(q, θ)
    if rand(rng) < exp(ℓw′ - ℓw)
        θ′, ℓπ′
    else
        θ, ℓπ
    end
end

function transition_rwmh(rng::Random.AbstractRNG, model, σ, θ::Real)
    q  = Normal(θ, σ)
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    if rand(rng) < exp(ℓπ′ - ℓπ)
        θ′, ℓπ′
    else
        θ, ℓπ
    end
end

function transition_rwmh(rng::Random.AbstractRNG, model, σ, θ::AbstractVector)
    q  = MvNormal(θ, σ)
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    if rand(rng) < exp(ℓπ′ - ℓπ)
        θ′, ℓπ′
    else
        θ, ℓπ
    end
end
