# ReversibleJump

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Red-Portal.github.io/ReversibleJump.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Red-Portal.github.io/ReversibleJump.jl/dev/)
[![Build Status](https://github.com/Red-Portal/ReversibleJump.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Red-Portal/ReversibleJump.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Red-Portal/ReversibleJump.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Red-Portal/ReversibleJump.jl)

A package providing reversible jump MCMC algorithms.
In particular, it provides the following implementations:
* The original algorithm by Green[^G1995],
* the annealed importance sampling jump proposals by Karagiannis and Andrieu[^KA2013], and
* the non-reversible jump proposals by Gagnon and Doucet[^GD2020].

For the types of jump moves, the package currently only provides birth and death.
But additional moves will also be added in the future.

## Example
An example of using the package is provided in `SinusoidDetection/`.
It contains a replication of the sinusoid joint estimation and detection model by Andrieu and Doucet [^AD1999].

## References
[^G1995]: Green, P. J. (1995). Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. Biometrika, 82(4), 711-732.
[^KA2013]: Karagiannis, G., & Andrieu, C. (2013). Annealed importance sampling reversible jump MCMC algorithms. Journal of Computational and Graphical Statistics, 22(3), 623-648.
[^GD2020]: Gagnon, P., & Doucet, A. (2020). Nonreversible jump algorithms for Bayesian nested model selection. Journal of Computational and Graphical Statistics, 30(2), 312-323.
[^AD1999]: Andrieu, C., & Doucet, A. (1999). Joint Bayesian model selection and estimation of noisy sinusoids via reversible jump MCMC. IEEE Transactions on Signal Processing, 47(10), 2667-2676.
