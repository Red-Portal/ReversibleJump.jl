
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
function mcmc_step             end

abstract type AbstractSampler end

abstract type AbstractJumpProposal end

struct RJState{Param, NT <: NamedTuple}
    param::Param
    lp   ::Real
    order::Int
    stat ::NT
end

abstract type AbstractMove end

struct Update <: AbstractMove end
struct Birth  <: AbstractMove end
struct Death  <: AbstractMove end

include("utils.jl")
include("proposal_ais.jl")
include("proposal_indep.jl")

#include("jump_move.jl")
#include("rjmcmc.jl")
#include("nrjmcmc.jl")
#include("sample.jl")

end
