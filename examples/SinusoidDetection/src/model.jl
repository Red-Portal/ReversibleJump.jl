
struct SinusoidModel{
    Y <: AbstractVector, F <: Real, P
} <: AbstractMCMC.AbstractModel
    y         ::Y
    gamma0    ::F
    nu0       ::F
    delta     ::F
    orderprior::P
end

function SinusoidModel(
    rng::Random.AbstractRNG, N::Int, gamma0::Real, nu0::Real, delta::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = delta*delta

    D   = spectrum_matrix(ω, N)
    DᵀD = PDMats.PDMat(Hermitian(D'*D) + 1e-15*I)
    y   = rand(rng, MvNormal(Zeros(N), σ²*(δ²*PDMats.X_invA_Xt(DᵀD, D) + I)))
    SinusoidModel(y, gamma0, nu0, delta, orderprior)
end

struct SinusoidUniformLocalProposal end

function spectrum_matrix(ω::AbstractVector, N::Int)
    k = length(ω)
    D = zeros(N, 2*k)
    for i in 1:N
        for j in 1:k
            D[i,2*j - 1] = cos(ω[j]*(i-1))
            D[i,2*j    ] = sin(ω[j]*(i-1))
        end
    end
    D
end

function ReversibleJump.logdensity(model::SinusoidModel, ω)
    @unpack y, gamma0, nu0, delta, orderprior = model

    N  = length(y)
    δ² = delta*delta
    k  = length(ω)

    ℓp_y = if k == 0
        (-(N + nu0)/2)*log(gamma0 + dot(y, y))
    else
        for j in 1:k
            if ω[j] > π || ω[j] < 0
                return -Inf
            end
        end
        D = spectrum_matrix(ω, N)
        try
            DᵀD = PDMats.PDMat(Hermitian(D'*D))
            P   = I - δ²/(1 + δ²)*PDMats.X_invA_Xt(DᵀD, D)
            (N + nu0)/-2*log(gamma0 + PDMats.quad(P, y)) - k*log(1 + δ²)
        catch
            return -Inf
        end
    end
    ℓp_k = logpdf(orderprior, k)
    ℓp_θ = k*logpdf(Uniform(0, π), π/2)
    ℓp_y + ℓp_k + ℓp_θ
end

function ReversibleJump.local_proposal_sample(
    rng::Random.AbstractRNG,
    ::SinusoidModel,
    ::SinusoidUniformLocalProposal
)
    rand(rng, Uniform(0, π))
end

function ReversibleJump.local_proposal_logpdf(
    ::SinusoidModel,
    ::SinusoidUniformLocalProposal,
    θ, j
)
    logpdf(Uniform(0, π), θ[j])
end

function ReversibleJump.local_insert(::SinusoidModel, θ, j, θj)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(::SinusoidModel, θ, j)
    deleteat!(copy(θ), j), θ[j]
end

struct IMHSinusoid <: AbstractMCMC.AbstractSampler
    n_snapshots::Int
end

function ReversibleJump.transition_mcmc(rng::Random.AbstractRNG, mcmc::IMHSinusoid, model, θ)
    σ    = 1/5/mcmc.n_snapshots
    q    = Uniform(0, π)
    θ    = copy(θ)
    k    = length(θ)
    idxs = randperm(rng, k) # the kernel is not reversible without random permutation
    for idx in idxs
        model_gibbs = GibbsObjective(model, idx, θ)
        θ′idx, _ = if rand(Bernoulli(0.2))
            transition_imh(rng, model_gibbs, q, θ[idx])
        else
            transition_rwmh(rng, model_gibbs, σ, θ[idx])
        end
        θ[idx]  = θ′idx
    end
    θ, logdensity(model, θ)
end
