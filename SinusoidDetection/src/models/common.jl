
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
    ν0::Real,
    γ0::Real,
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

function sample_amplitude_and_noise(
    rng::Random.AbstractRNG,
    y  ::AbstractVector,
    D  ::AbstractMatrix,
    DᵀD::PDMats.AbstractPDMat,
    ν0 ::Real,
    γ0 ::Real,
    δ² ::Real,
)
    N    = length(y)
    M⁻¹  = (1 + 1/δ²)*DᵀD
    P    = I - δ²/(1 + δ²)*PDMats.X_invA_Xt(DᵀD, D)
    m    = M⁻¹\(D*y)
    yᵀPy = PDMats.quad(P, y)
    σ²   = rand(rng, InverseGamma((ν0 + N)/2, (γ0 + yᵀPy)/2));
    a    = PDMats.whiten(M⁻¹, randn(rng, N)) + m 
    a, σ²
end

function sample_gibbs_snr(
    rng ::Random.AbstractRNG,
    y   ::AbstractVector,
    ω   ::AbstractMatrix,
    ν0  ::Real,
    γ0  ::Real,
    α_δ²::Real,
    β_δ²::Real,
    δ²  ::Real,
)
    k     = length(ω)
    N     = length(y)
    D     = spectrum_matrix(ω, N)
    DᵀD   = PDMats.PDMat(Hermitian(D'*D))
    a, σ² = sample_amplitude_and_noise(rng, y, D, DᵀD, ν0, γ0, δ²)
    s²    = PDMats.quad(DᵀD, a)/2/σ²
    rand(rng, InverseGamma(k + α_δ², s² + β_δ²))
end
