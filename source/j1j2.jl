include("pvqd.jl")
include("ansatz.jl")

using JLD2

function j1j2_pvqd(
    spins,
    edges,
    g,
    dt,
    n_steps,
    depth,
    cut_trotter,
    n_shots,
    initial_params,
    optimizer_args::OptimizerArgs,
    initial_delta,
    restrict_entanglement,
    entanglement_args,
    rotations,
)

    j1 = j2 = 999
    for (_, j) in edges
        if j1 == 999
            j1 = j
        elseif j2 == 999 && j != j1
            j2 = j
        end
    end
    if j1 < j2
        j1, j2 = j2, j1
    end

    ansatz, indices = graph_ansatz(
        spins,
        edges,
        Dict("j" => j1, "J" => j2),
        depth,
        entanglement_args.gate_indices,
        initial_params,
        rotations = rotations,
        return_param_types = false,
    )
    print(j1, j2)
    loss = :local
    guess_strategy = :delta


    if cut_trotter
        trotter_step = graph_trotter(spins, edges, g, dt)
        cut_overhead = sqrt(gamma(2 * j2 * dt))^2
    else
        cut_edges = [j == j1 ? (edge, j) : (edge, 0) for (edge, j) in edges]
        trotter_step = graph_trotter(spins, cut_edges, g, dt)
        cut_overhead = 1.0
    end


    return PVQD(
        ansatz,
        n_steps,
        initial_params,
        trotter_step,
        optimizer_args,
        restrict_entanglement,
        entanglement_args,
        indices,
        loss,
        n_shots,
        guess_strategy,
        initial_delta,
        cut_overhead,
    )

end


begin
    num = "203"
    j1 = 1
    j2 = 0.25
    g = 1
    spins = [1, 2, 3, 4]
    edges = [
        # horizontal
        ((1, 2), j1),
        ((3, 4), j1),
        # vertical
        ((1, 3), j1),
        ((2, 4), j1),
        # diagonal
        ((1, 4), j2),
        ((2, 3), j2),
    ]
    dt = 0.05
    n_steps = 40
    depth = 2

    restrict_entanglement = true
    cross_gates = [true for _ = 1:depth]
    ent_args = EntanglementArgs(cross_gates, Inf, 0.001, true, true)
    rotations = :xx
    n_shots = nothing
    single_q_params = rotations == :full ? 3 : 1
    #initial_params = zeros(4 * (depth) + 4 * single_q_params * (depth + 1))
    initial_params =
        zeros(depth * length(edges) + (depth + 1) * length(spins) * single_q_params)
    optimizer_args = OptimizerArgs(
        1e-3,
        nothing,
        nothing,
        nothing,
        nothing,
        5,
        500,
        :adam,
        (0.9, 0.999),
        :exact,
        pi / 2,
    )
    initial_delta = 0.00


    cut_trotter = true
    pvqd = j1j2_pvqd(
        spins,
        edges,
        g,
        dt,
        n_steps,
        depth,
        cut_trotter,
        n_shots,
        initial_params,
        optimizer_args,
        initial_delta,
        restrict_entanglement,
        ent_args,
        rotations,
    )

    args = Dict(
        "j1" => j1,
        "j2" => j2,
        "g" => g,
        "dt" => dt,
        "n_steps" => n_steps,
        "depth" => depth,
        "cut_trotter" => cut_trotter,
        "n_shots" => n_shots,
        "initial_params" => initial_params,
        "optimizer_args" => optimizer_args,
        "initial_delta" => initial_delta,
        "restrict_entanglement" => restrict_entanglement,
        "ent_args" => ent_args,
        "rotations" => rotations,
        "spins" => spins,
        "edges" => edges,
    )

    try
        load_object("j1j2_results/$(num)/args")
        println("Experiment already exists.")
    catch
        result = run(pvqd)
        mkdir("j1j2_results/$(num)")
        save_object("j1j2_results/$(num)/args", args)
        save_object("j1j2_results/$(num)/result", result)
    end
end

include("plot.jl")


# Rename results to match pattern used for plotting
main_num = 200
nums = [200, 201, 202, 203] # numbers of experiments to bundle under main_num
for num in nums
    args = load_object("j1j2_results/$(num)/args")
    τ = args["ent_args"].threshold
    if sum(args["ent_args"].gate_indices) == 0
        if args["cut_trotter"]
            τ = 1.0
        else
            τ = -1.0
        end
    end
    try
        mkdir("j1j2_results/20230717_$(main_num)_tau_$(τ)_")
    catch
    end
    println("j1j2_results/20230717_$(main_num)_tau_$(τ)_")
    result = load_object("j1j2_results/$(num)/result")
    save_object("j1j2_results/20230717_$(main_num)_tau_$(τ)_/setup_args.txt", args)
    save_object("j1j2_results/20230717_$(main_num)_tau_$(τ)_/result", result)
end
