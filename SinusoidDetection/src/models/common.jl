
struct SinusoidLocalProposal end

function ReversibleJump.local_proposal_sample(
    rng  ::Random.AbstractRNG,
    model::AbstractSinusoidModel,
         ::SinusoidLocalProposal
)
    rand(rng, model.freqprop)
end

function spectrum_matrix(ω::AbstractVector, N::Int)
    k = length(ω)
    D = zeros(eltype(ω), N, 2*k)
    @turbo for i in 1:N
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

function spectrum_energy_proposal(y::AbstractVector)
    n_snap   = length(y)
    n_fft    = nextpow(2, n_snap)*2
    n_fft_in = n_fft ÷ 2
    bin_uniform = map(0:n_fft_in-1) do l
        Uniform(l*2*π/n_fft, (l+1)*2*π/n_fft)
    end
    y_pad = vcat(y, zeros(n_fft - n_snap))
    Y     = fft(y_pad)[1:n_fft_in]
    mag_Y = abs2.(Y)

    # Clamp the unnormalized proposal density so that
    # IMH importance weights are bounded.
    peak_power   = maximum(mag_Y)
    p_bin_unnorm = max.(mag_Y, peak_power*0.01)
    p_bin        = p_bin_unnorm / sum(p_bin_unnorm)

    MixtureModel(bin_uniform, p_bin)
end
