
module ReversibleJump

export
    local_proposal_sample,
    local_proposal_logpdf,
    local_insert,
    local_deleteat,
    transition_mcmc,
    logdensity,
    BirthDeath,
    IndepJumpProposal,
    AnnealedJumpProposal,
    QPath,
    GeometricPath,
    ArithmeticPath,
    ReversibleJumpMCMC,
    NonReversibleJumpMCMC,
    sample

using AbstractMCMC
using Accessors
using Distributions
using LogExpFunctions
using OnlineStats
using ProgressMeter
using Random
using SimpleUnPack
using StatsBase

function local_proposal_sample end
function local_proposal_logpdf end
function local_insert          end
function local_deleteat        end
function logdensity            end
function propose_jump          end
function transition_mcmc       end
function transition_jump       end

abstract type AbstractRJMCMCSampler <: AbstractMCMC.AbstractSampler end

abstract type AbstractJumpProposal end

abstract type AbstractRJState end

struct RJState{Param, NT <: NamedTuple} <: AbstractRJState
    param::Param
    lp   ::Real
    order::Int
    stats::NT
end

struct NRJState{Param, NT <: NamedTuple} <: AbstractRJState
    direction::Bool
    param    ::Param
    lp       ::Real
    order    ::Int
    stats    ::NT
end

abstract type AbstractJumpMove     end

abstract type AbstractJumpMovePair end

struct IndepJumpProposal{Prop} <: AbstractJumpProposal
    local_proposal::Prop
end

include("utils.jl")
include("ais.jl")
include("birthdeath.jl")
include("jump.jl")
include("rjmcmc.jl")
include("nrjmcmc.jl")
include("sample.jl")
include("modelposterior.jl")

end
