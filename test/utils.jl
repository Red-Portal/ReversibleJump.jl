
function tv_distance(
    p::DiscreteDistribution,
    q::DiscreteDistribution
)
    # d_{TV}(P,Q) = 1 - (P ∧ Q) = 1 - ∫min(p, q)dμ

    sup    = union(support(p), support(q))
    ℓp     = map(Base.Fix1(logpdf, p), sup)
    ℓq     = map(Base.Fix1(logpdf, q), sup)
    logdtv = log1mexp(logsumexp(min.(ℓp, ℓq)))
    exp(logdtv)
end
