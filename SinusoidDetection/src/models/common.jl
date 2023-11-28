
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
            D    = spectrum_matrix(ω, N)
            DᵀD  = PDMats.PDMat(Hermitian(D'*D + 1e-10*I))
            Dᵀy  = D'*y
            yᵀPy = dot(y, y) - δ²/(1 + δ²)*PDMats.invquad(DᵀD, Dᵀy)
            (N + ν0)/-2*log(γ0 + yᵀPy) - k*log(1 + δ²)
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
    Dᵀy  = D'*y
    m    = M⁻¹\Dᵀy
    yᵀPy = dot(y, y) - PDMats.invquad(M⁻¹, Dᵀy)
    yᵀPy = max(yᵀPy, eps(eltype(y)))
    σ²   = rand(rng, InverseGamma((ν0 + N)/2, (γ0 + yᵀPy)/2))
    a    = rand(rng, MvNormal(m, σ²*inv(M⁻¹)))
    a, σ²
end

function sample_gibbs_snr(
    rng ::Random.AbstractRNG,
    y   ::AbstractVector,
    ω   ::AbstractVector,
    ν0  ::Real,
    γ0  ::Real,
    α_δ²::Real,
    β_δ²::Real,
    δ²  ::Real,
)
    k     = length(ω)
    N     = length(y)
    D     = spectrum_matrix(ω, N)
    DᵀD   = PDMats.PDMat(Hermitian(D'*D + 1e-10*I))
    a, σ² = sample_amplitude_and_noise(rng, y, D, DᵀD, ν0, γ0, δ²)
    rand(rng, InverseGamma(k + α_δ², PDMats.quad(DᵀD, a)/2/σ² + β_δ²))
end

function sample_signal(
    rng::Random.AbstractRNG, ω::AbstractVector, N::Int, σ²::Real, δ²::Real
)
    D   = spectrum_matrix(ω, N)
    DᵀD = PDMats.PDMat(Hermitian(D'*D) + 1e-10*I)
    rand(rng, MvNormal(Zeros(N), σ²*(δ²*PDMats.X_invA_Xt(DᵀD, D) + I)))
end
