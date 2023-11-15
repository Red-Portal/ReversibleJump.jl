
struct ConstantLocalProposal end

struct DiscreteModel{C <: Distributions.DiscreteDistribution} <: AbstractMCMC.AbstractModel
    order_dist::C
end

function ReversibleJump.logdensity(model::DiscreteModel, θ)
    logpdf(model.order_dist, length(θ))
end

function ReversibleJump.local_proposal_sample(
    ::Random.AbstractRNG,
    ::DiscreteModel,
    ::ConstantLocalProposal
)
    1.0
end

function ReversibleJump.local_proposal_logpdf(
    ::DiscreteModel,
    ::ConstantLocalProposal,
    θ, j
)
    0.0
end

function ReversibleJump.local_insert(::DiscreteModel, θ, j, θj)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(::DiscreteModel, θ, j)
    deleteat!(copy(θ), j), θ[j]
end

struct IdentityKernel <: AbstractMCMC.AbstractSampler end

function ReversibleJump.transition_mcmc(::Random.AbstractRNG, ::IdentityKernel, model, θ)
    copy(θ), ReversibleJump.logdensity(model, θ)
end

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::DiscreteModel)
    k = rand(rng, model.order_dist)
    zeros(k), nothing
end
