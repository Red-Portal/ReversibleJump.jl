
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

function marginal_covariance_residual(ω, N, δ²)
    D   = spectrum_matrix(ω, N)
    DᵀD = PDMats.PDMat(Hermitian(D'*D))
    I - δ²/(1 + δ²)*PDMats.X_invA_Xt(DᵀD, D)
end

function collapsed_likelihood(
    y ::AbstractVector,
    ω ::AbstractVector,
    δ²::Real,
    γ0::Real,
    ν0::Real
)
    N  = length(y)
    k  = length(ω)
    if k == 0
        N = length(y)
        (-(N + ν0)/2)*log(γ0 + dot(y, y))
    else
        for j in 1:k
            if ω[j] > π || ω[j] < 0
                return -Inf
            end
        end
        try
            P = marginal_covariance_residual(ω, N, δ²)
            (N + ν0)/-2*log(γ0 + PDMats.quad(P, y)) - k*log(1 + δ²)
        catch
            return -Inf
        end
    end
end
