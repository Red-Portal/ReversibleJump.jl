
function pm_next!(pm, stat::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stat)])
end

struct OnlineMean{T <: Real} end

init(::OnlineMean{T}) where {T} = (zero(T), 0)

function fit(::OnlineMean, state, value)
    μ_prev = first(state)
    n      = last(state)
    μ_next = μ_prev*n/(n + 1) + value/(n + 1)
    return μ_next, (μ_next, n + 1)
end
