
module ReversibleJump

export
    local_proposal_sample,
    local_proposal_logpdf,
    local_insert,
    local_deleteat,
    model_order,
    logdensity,
    Birth,
    Death,
    IndepJumpProposal,
    AnnealedJumpProposal,
    GeometricPath,
    ArithmeticPath

using Accessors
using Distributions
using LogExpFunctions
using OnlineStats
using ProgressMeter
using Random
using SimpleUnPack

function local_proposal_sample end
function local_proposal_logpdf end
function local_insert          end
function local_deleteat        end
function logdensity            end
function model_order           end
function propose_jump          end
function step_mcmc             end
function step_jump             end

abstract type AbstractSampler end

abstract type AbstractJumpProposal end

struct RJState{Param, NT <: NamedTuple}
    param::Param
    lp   ::Real
    order::Int
    stats::NT
end

abstract type AbstractJumpMove end

struct IndepJumpProposal{Prop} <: AbstractJumpProposal
    local_proposal::Prop
end

include("utils.jl")
include("ais.jl")
include("birthdeath.jl")
include("jump.jl")

#include("jump_move.jl")
#include("rjmcmc.jl")
#include("nrjmcmc.jl")
#include("sample.jl")

end
