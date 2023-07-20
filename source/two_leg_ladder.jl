using Dates, Random, JLD2, Plots
using Base.Threads
using Profile

include("sketch.jl")
include("pvqd.jl")

"Test for a 2D hardware-efficient experiment."

sketch_string = """
g j g J g j g
j - j - j - j 
g j g J g j g
"""

j1 = 1.0
j2 = 0.25
g = 1.0

sk = sketch(sketch_string; j1 = j1, j2 = j2, g = g)

spins = sk.n_spins
j_count = count("j", sketch_string)
J_count = count("J", sketch_string)


# Setup
begin
    d = 5
    rotations = :xx
    dt = 0.05
    n_steps = 3
    lr = 1e-3
    abstol = nothing
    gradtol = nothing
    rel_ftol = nothing
    rel_gtol = nothing
    bin_size = 5
    maxiter = 100

    restrict_entanglement = true
    reset_momentum = true
    reset_v = false
    indices = [true for _ = 1:d]
    adam_args = (0.9, 0.999)
    initial_delta = 0.0
    loss = :local
    shots = nothing
    backstep_size = 0.001

    guess_strategy = :delta

    opt_args = OptimizerArgs(
        lr,
        abstol,
        gradtol,
        rel_ftol,
        rel_gtol,
        bin_size,
        maxiter,
        :adam,
        adam_args,
        :exact,
        pi / 2,
    )

end

thresholds = [-1, 1, 100, 1000, Inf]
seeds = 1:1:1
paths = []
labels = []

num = "xxx"
@threads for thresh in thresholds
    for seed in seeds
        experiment_num = "$(num)_tau_$(thresh)_seed_$seed"
        date = Dates.format(now(), "yyyyddmm")
        path = "2dresults/$(date)_$experiment_num"
        push!(paths, path)


        if thresh == Inf
            # infinite entanglement, full ansatz
            initial_params = zeros((d + 1) * spins + d * (j_count + J_count))
            restrict_entanglement = true
            entanglement_args =
                EntanglementArgs(indices, thresh, backstep_size, reset_momentum, reset_v)
            push!(labels, "threshold = $thresh")
            cut_trotter = true

        elseif thresh > 1
            # finite entanglement
            initial_params = zeros((d + 1) * spins + d * (j_count + J_count))
            restrict_entanglement = true
            entanglement_args =
                EntanglementArgs(indices, thresh, backstep_size, reset_momentum, reset_v)
            push!(labels, "threshold = $thresh")
            cut_trotter = true
        elseif thresh < 0
            # pure mean-field, no entangling trotter step
            initial_params = zeros((d + 1) * spins + d * j_count)
            entanglement_args = EntanglementArgs(
                [false for i = 1:d],
                100000,
                backstep_size,
                reset_momentum,
                reset_v,
            )
            restrict_entanglement = true
            cut_trotter = false
            push!(labels, "No entanglement.")
        else
            # mean-field, only entangling trotter step
            restrict_entanglement = true
            initial_params = zeros((d + 1) * spins + d * j_count)
            entanglement_args = EntanglementArgs(
                [false for i = 1:d],
                100000,
                backstep_size,
                reset_momentum,
                reset_v,
            )
            push!(labels, "Only Trotter entanglement.")
            cut_trotter = true

        end

        Random.seed!(seed)

        setup_args = Dict(
            "sketch" => sk,
            "d" => d,
            "rotations" => rotations,
            "dt" => dt,
            "n_steps" => n_steps,
            "opt_args" => opt_args,
            "initial_params" => initial_params,
            "restrict_entanglement" => restrict_entanglement,
            "entanglement_args" => entanglement_args,
            "loss" => loss,
            "shots" => shots,
            "seed" => seed,
            "guess_strategy" => guess_strategy,
            "initial_delta" => initial_delta,
            "cut_trotter" => cut_trotter,
        )

        try
            args = load_object(path * "/setup_args.txt")
            # check whether all args match
            for (key, value) in setup_args
                if args[key] != value
                    print(
                        "Found experiment with different setup args. Please change the experiment number.",
                    )
                end
            end
            print("Found matching experiment already.")
        catch
            mkpath(path)
            print("New experiment running")
            pvqd = PVQD(
                sk,
                depth = d,
                rotations = rotations,
                n_steps = n_steps,
                dt = dt,
                initial_params = initial_params,
                optimizer_args = opt_args,
                restrict_entanglement = restrict_entanglement,
                entanglement_args = entanglement_args,
                loss = loss,
                n_shots = shots,
                guess_strategy = guess_strategy,
                initial_delta = initial_delta,
                cut_trotter = cut_trotter,
            )

            result = run(pvqd)
            save_object(path * "/setup_args.txt", setup_args)
            save_object(path * "/result", result)
            println("Experiment $experiment_num finished. Saved to $path")
            println(
                "Tolerance: $abstol, Threshold: $thresh, Final fidelity: $(result.fidelities[end])",
            )
        end
    end
end
