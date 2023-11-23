using ReversibleJump
using Documenter

DocMeta.setdocmeta!(ReversibleJump, :DocTestSetup, :(using ReversibleJump); recursive=true)

makedocs(;
    modules=[ReversibleJump],
    repo="https://github.com/Red-Portal/ReversibleJump.jl/blob/{commit}{path}#{line}",
    sitename="ReversibleJump.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Red-Portal.github.io/ReversibleJump.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"       => "index.md",
        "Benchmarks" => "benchmarks.md"
    ],
)

deploydocs(;
    repo="github.com/Red-Portal/ReversibleJump.jl",
    devbranch="main",
)
