
@testset "modelposterior" begin
    prior  = truncated(Geometric(0.2); lower=1, upper=10)
    model  = DiscreteProductModel(
        truncated(Poisson(5); lower=1, upper=10), prior
    )

    n_anneal   = 8
    path       = ArithmeticPath(n_anneal)
    prop       = ConstantLocalProposal()
    mcmc       = IdentityKernel()
    n_samples  = 1000

    initial_order  = 1
    initial_params = [0.0]

    rjmcmc = ReversibleJump.ReversibleJumpMCMC(
        prior, AnnealedJumpProposal(prop, path), mcmc
    )
    samples, stats = ReversibleJump.sample(
        rjmcmc,
        model,
        n_samples,
        initial_order,
        initial_params,
        show_progress=false
    )
    emp_order_post  = fit(Categorical, [stat.order for stat in stats])
    true_order_post = posterior(model)
    rb_order_post   = ReversibleJump.modelposterior(stats, prior)

    @test tv_distance(emp_order_post, true_order_post) >
        tv_distance(rb_order_post, true_order_post)
end
