
struct SinusoidUniformLocalProposal end

function ReversibleJump.local_proposal_sample(
    rng::Random.AbstractRNG,
    ::AbstractSinusoidModel,
    ::SinusoidUniformLocalProposal
)
    rand(rng, Uniform(0, π))
end

function ReversibleJump.local_proposal_logpdf(
    ::AbstractSinusoidModel,
    ::SinusoidUniformLocalProposal,
    θ, j
)
    logpdf(Uniform(0, π), θ[j])
end

function ReversibleJump.local_insert(::AbstractSinusoidModel, θ, j, θj)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(::AbstractSinusoidModel, θ, j)
    deleteat!(copy(θ), j), θ[j]
end

ReversibleJump.transition_mcmc(
    rng  ::Random.AbstractRNG,
    mcmc ::AbstractSliceSampling,
    model,
    θ
) = slice_sampling(rng, mcmc, model, θ)

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

function collapsed_likelihood(
    y ::AbstractVector,
    ω ::AbstractVector,
    δ ::Real,
    γ0::Real,
    ν0::Real
)
    N  = length(y)
    k  = length(ω)
    if k == 0
        N = length(y)
        (-(N + ν0)/2)*log(γ0 + dot(y, y))
    else
        δ² = δ*δ
        for j in 1:k
            if ω[j] > π || ω[j] < 0
                return -Inf
            end
        end
        D = spectrum_matrix(ω, N)
        try
            DᵀD = PDMats.PDMat(Hermitian(D'*D))
            P   = I - δ²/(1 + δ²)*PDMats.X_invA_Xt(DᵀD, D)
            (N + ν0)/-2*log(γ0 + PDMats.quad(P, y)) - k*log(1 + δ²)
        catch
            return -Inf
        end
    end
end
