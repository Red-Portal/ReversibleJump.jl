
function rand_sinusoids_knownsnr(
    rng::Random.AbstractRNG, N::Int, nu0::Real, gamma0::Real, delta2::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    y  = sample_signal(rng, ω, N, σ², delta2)
    SinusoidKnownSNR(y, nu0, gamma0, delta2, orderprior)
end

function rand_sinusoids_knownsnr(
    N::Int, nu0::Real, gamma0::Real, delta2::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    rand_sinusoids_knownsnr(Random.default_rng(), N, nu0, gamma0, delta2, orderprior)
end

function rand_sinusoids_unknownsnr(
    rng         ::Random.AbstractRNG,
    N           ::Int,
    nu0         ::Real,
    gamma0      ::Real,
    alpha_delta2::Real,
    beta_delta2 ::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    k  = rand(rng, orderprior)
    ω  = rand(rng, Uniform(0, π), k)
    σ² = rand(rng, InverseGamma(nu0/2, gamma0/2))
    δ² = rand(rng, InverseGamma(alpha_delta2, beta_delta2))
    δ  = sqrt(δ²)
    y  = sample_signal(rng, ω, N, σ², δ)
    SinusoidUnknownSNR(y, nu0, gamma0, alpha_delta2, beta_delta2, orderprior)
end

function rand_sinusoids_unknownsnr(
    N           ::Int,
    nu0         ::Real,
    gamma0      ::Real,
    alpha_delta2::Real,
    beta_delta2 ::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    rand_sinusoids_unknownsnr(
        Random.default_rng(),
        N,
        nu0,
        gamma0,
        alpha_delta2,
        beta_delta2,
        orderprior
    )
end

function rand_sinusoids_unknownsnr_reparam(
    rng         ::Random.AbstractRNG,
    N           ::Int,
    nu0         ::Real,
    gamma0      ::Real,
    alpha_delta2::Real,
    beta_delta2 ::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    model_base = rand_sinusoids_unknownsnr(
        rng, N, nu0, gamma0, alpha_delta2, beta_delta2, orderprior
    )
    SinusoidUnknownSNRReparam(
        model_base.y,
        model_base.nu0,
        model_base.gamma0,
        model_base.alpha_delta2,
        model_base.beta_delta2,
        model_base.orderprior,
    )
end

function rand_sinusoids_unknownsnr_reparam(
    N           ::Int,
    nu0         ::Real,
    gamma0      ::Real,
    alpha_delta2::Real,
    beta_delta2 ::Real,
    orderprior = truncated(Poisson(3), upper=floor(Int, (N-1)/2))
)
    rand_sinusoids_unknownsnr_reparam(
        Random.default_rng(),
        N,
        nu0,
        gamma0,
        alpha_delta2,
        beta_delta2,
        orderprior
    )
end
