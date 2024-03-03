
@testset "sample" begin
    rng    = Random.default_rng()
    model  = DiscreteModel(Poisson(4))

    n_anneal   = 8
    path       = ArithmeticPath(n_anneal)
    prop       = ConstantLocalProposal()
    mcmc       = IdentityKernel()
    n_samples  = 100

    initial_order  = 1
    initial_params = [0.0]

    @testset for sampler in [
        ReversibleJump.ReversibleJumpMCMC(
            model.order_dist, AnnealedJumpProposal(prop, path), mcmc
        ),
        ReversibleJump.ReversibleJumpMCMC(
            model.order_dist, IndepJumpProposal(prop), mcmc
        ),
        ReversibleJump.NonReversibleJumpMCMC(
            AnnealedJumpProposal(prop, path), mcmc
        ),
        ReversibleJump.NonReversibleJumpMCMC(
            IndepJumpProposal(prop), mcmc
        ),
    ]
        @testset "AbstractMCMC.sample" begin
            samples = AbstractMCMC.sample(
                model,
                sampler,
                n_samples;
                initial_order,
                initial_params,
                progress=false
            )
            @test length(samples) == n_samples
        end

        @testset "sample" begin
            samples, stats = ReversibleJump.sample(
                sampler,
                model,
                n_samples,
                initial_order,
                initial_params;
                show_progress=false
            )
            @test length(samples) == n_samples
            @test length(stats)   == n_samples
        end

        @testset "sample custom rng" begin
            samples, stats = ReversibleJump.sample(
                rng,
                sampler,
                model,
                n_samples,
                initial_order,
                initial_params;
                show_progress=false
            )
            @test length(samples) == n_samples
            @test length(stats)   == n_samples
        end

        @testset "sample custom callback" begin
            callback(param, stats) = (foo = 42,)
            samples, stats = ReversibleJump.sample(
                sampler,
                model,
                n_samples,
                initial_order,
                initial_params;
                show_progress=false, callback=callback
            )
            @test length(samples) == n_samples
            @test length(stats)   == n_samples
            @test all([stat.foo == 42 for stat in stats])
        end
    end
end
