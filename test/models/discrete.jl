
struct ConstantLocalProposal end

struct DiscreteModel{
    C <: Distributions.DiscreteDistribution
} <: AbstractMCMC.AbstractModel
    order_dist::C
end

function ReversibleJump.logdensity(model::DiscreteModel, θ)
    logpdf(model.order_dist, length(θ))
end

struct DiscreteProductModel{
    C1 <: Distributions.DiscreteDistribution,
    C2 <: Distributions.DiscreteDistribution
} <: AbstractMCMC.AbstractModel
    order_dist1::C1
    order_dist2::C2
end

function ReversibleJump.logdensity(model::DiscreteProductModel, θ)
    logpdf(model.order_dist1, length(θ)) +
        logpdf(model.order_dist2, length(θ))
end

function posterior(
    model::DiscreteProductModel{C1, C2}
) where {C1 <: Truncated, C2 <: Truncated}
    sup       = union(support(model.order_dist1), support(model.order_dist2))
    ℓp_unnorm = map(Base.Fix1(ReversibleJump.logdensity, model), map(zeros, sup))
    ℓZ        = logsumexp(ℓp_unnorm)
    p         = @. exp(ℓp_unnorm - ℓZ)
    p         = p/sum(p)
    DiscreteNonParametric{eltype(sup),eltype(p),typeof(sup),typeof(p)}(sup, p)
end

function ReversibleJump.local_proposal_sample(
    ::Random.AbstractRNG,
    ::Union{DiscreteModel, DiscreteProductModel},
    ::ConstantLocalProposal
)
    1.0
end

function ReversibleJump.local_proposal_logpdf(
    ::Union{DiscreteModel, DiscreteProductModel},
    ::ConstantLocalProposal,
    θ, j
)
    0.0
end

function ReversibleJump.local_insert(
    ::Union{DiscreteModel, DiscreteProductModel}, θ, j, θj
)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(
    ::Union{DiscreteModel, DiscreteProductModel}, θ, j
)
    deleteat!(copy(θ), j), θ[j]
end

struct IdentityKernel <: AbstractMCMC.AbstractSampler end

function ReversibleJump.transition_mcmc(
    ::Random.AbstractRNG, ::IdentityKernel, model, θ
)
    copy(θ), ReversibleJump.logdensity(model, θ)
end

function MCMCTesting.sample_joint(
    rng::Random.AbstractRNG, model::Union{DiscreteModel, DiscreteProductModel}
)
    k = rand(rng, model.order_dist)
    zeros(k), nothing
end
