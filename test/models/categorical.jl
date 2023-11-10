
struct ConstantLocalProposal end

struct CategoricalModelSpace{C <: Distributions.Categorical, F <: Real}
    order_prob    ::C
    log_normalizer::F
end

function ReversibleJump.logdensity(model::CategoricalModelSpace, θ)
    order_prob     = model.order_prob
    log_normalizer = model.log_normalizer

    if length(θ) + 1 ≤ ncategories(order_prob)
        logpdf(order_prob, length(θ) + 1) + log_normalizer
    else
        -Inf
    end
end

function ReversibleJump.model_order(::CategoricalModelSpace, θ)
    length(θ)
end

function ReversibleJump.local_proposal_sample(
    ::Random.AbstractRNG,
    ::CategoricalModelSpace,
    ::ConstantLocalProposal
)
    1.0
end

function ReversibleJump.local_proposal_logpdf(
    ::CategoricalModelSpace,
    ::ConstantLocalProposal,
    θ, j
)
    0.0
end

function ReversibleJump.local_insert(::CategoricalModelSpace, θ, j, θj)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(::CategoricalModelSpace, θ, j)
    deleteat!(copy(θ), j), θ[j]
end

struct IdentityKernel <: AbstractMCMC.AbstractSampler end

function ReversibleJump.step_mcmc(::Random.AbstractRNG, ::IdentityKernel, model, θ)
    copy(θ), ReversibleJump.logdensity(model, θ)
end
