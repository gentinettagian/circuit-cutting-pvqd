using Dates
using JLD2
using Random
using YaoPlots, Plots
using CSV
using Base.Threads
include("ansatz.jl")
include("pvqd.jl")

# Setup
begin
    n_spins = 2
    n_blocks = 3
    d = 3 # depth of ansatz
    rotations = :xx # type of single qubit gates
    J1 = 1
    J2 = 0.25
    g = 1
    dt = 0.05
    n_steps = 40
    lr = 1e-2
    # Termination criteria
    abstol = nothing
    gradtol = nothing
    rel_ftol = nothing
    rel_gtol = nothing
    bin_size = 5
    maxiter = 200

    restrict_entanglement = true # whether to use threshold
    reset_momentum = true
    reset_v = false
    indices = [true for _ = 1:d] # which layers contain entangling gates
    adam_args = (0.9, 0.999)
    initial_delta = 0.00
    loss = :local
    shots = 5000
    #shots = nothing
    backstep_size = 0.00001
    guess_strategy = :delta


    shift = pi / 2
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
        :param_shift, # change to :param_shift for shot based experiments
        shift,
    )

end

thresholds = [-1.0, 1.0, 100, 1000, Inf]#, 5, 10, 25, 50, 250, 500, 5000]
seeds = 1:1:10
paths = []
labels = []

num = "018"
@threads for thresh in thresholds
    @threads for seed in seeds
        experiment_num = "$(num)_tau_$(thresh)_seed_$seed"
        #experiment_num = "$(num)_tau_$(thresh)"
        date = Dates.format(now(), "yyyymmdd")
        #path = "results/exp_thresh_1002/$(date)_$experiment_num"
        path = "results/$(date)_$experiment_num"
        push!(paths, path)

        if thresh == Inf
            initial_params = zeros(
                (d + 1) * (n_blocks * n_spins) +
                d * n_blocks * (n_spins - 1) +
                min(sum(indices), d) * (n_blocks - 1),
            )
            restrict_entanglement = true
            entanglement_args =
                EntanglementArgs(indices, thresh, backstep_size, reset_momentum, reset_v)
            push!(labels, "threshold = $thresh")
            cut_trotter = true

        elseif thresh > 1e6
            # infinite entanglement, full ansatz
            initial_params = zeros(
                (d + 1) * (n_blocks * n_spins) +
                d * n_blocks * (n_spins - 1) +
                min(sum(indices), d) * (n_blocks - 1),
            )
            restrict_entanglement = false
            entanglement_args = nothing
            cut_trotter = true

            push!(labels, "threshold = $thresh")
        elseif thresh > 1
            # finite entanglement
            initial_params = zeros(
                (d + 1) * (n_blocks * n_spins) +
                d * n_blocks * (n_spins - 1) +
                min(sum(indices), d) * (n_blocks - 1),
            )
            restrict_entanglement = true
            entanglement_args =
                EntanglementArgs(indices, thresh, backstep_size, reset_momentum, reset_v)
            push!(labels, "threshold = $thresh")
            cut_trotter = true
        elseif thresh < 0
            # pure mean-field, no entangling trotter step
            initial_params =
                zeros((d + 1) * (n_blocks * n_spins) + d * n_blocks * (n_spins - 1))
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
            initial_params =
                zeros((d + 1) * (n_blocks * n_spins) + d * n_blocks * (n_spins - 1))
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
            "n_spins" => n_spins,
            "n_blocks" => n_blocks,
            "d" => d,
            "rotations" => rotations,
            "J1" => J1,
            "J2" => J2,
            "g" => g,
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
            "shift" => shift,
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
                n_spins,
                n_blocks,
                d,
                rotations,
                J1,
                J2,
                g,
                n_steps,
                dt,
                initial_params,
                opt_args,
                restrict_entanglement,
                entanglement_args,
                loss,
                shots,
                guess_strategy,
                initial_delta,
                cut_trotter,
            )

            result = run(pvqd)
            save_object(path * "/setup_args.txt", setup_args)
            save_object(path * "/result", result)
            println("Experiment $experiment_num finished. Saved to $path")

        end
    end
end

