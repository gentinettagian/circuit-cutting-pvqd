
include("load_results.jl")
include("plot.jl")
include("plot.jl")

using Plots

using LinearAlgebra


"""Split a state into two blocks and get the entanglement entropy between them.
The `state` is a vector in C^2^n_qubits, and the `block_1` and `block_2` are
arrays of the indices of the qubits in the blocks.
"""
function entanglement_entropy(state, block_1::Array{Int,1}, block_2::Array{Int,1})
    n_qubits = length(block_1) + length(block_2)
    if n_qubits != log2(length(state))
        error(
            "The number of qubits in the blocks does not match the number of qubits in the state.",
            "Is currently $n_qubits, should be $(log2(length(state)))",
        )
    end
    reshaped_state = zeros(Complex, 2^length(block_1), 2^length(block_2))
    for (index, amplitude) in enumerate(state)
        bits = reverse(bitstring(index - 1)[end-n_qubits+1:end])
        bits_1 = ""
        bits_2 = ""
        for (i, bit) in enumerate(bits)
            if i in block_1
                bits_1 *= bit
            else
                bits_2 *= bit
            end
        end
        bits_1 = reverse(bits_1)
        bits_2 = reverse(bits_2)
        reshaped_state[parse(Int, bits_1, base = 2)+1, parse(Int, bits_2, base = 2)+1] =
            amplitude
    end
    _, s, _ = svd(reshaped_state)
    #println(s, sum(s .^ 2))
    s = s[s.>0]
    entropy = -sum(s .^ 2 .* log.(s .^ 2))

    return entropy
end


"Get the entanglement entropy of the states of a PVQD experiment"
function get_entropies(result, args, block_1, block_2)
    states = pvqd_states(result, get_ansatz(args))
    entropies = [entanglement_entropy(state, block_1, block_2) for state in states]
    return entropies
end



begin
    num = "805"
    results_loc = "results"
    thresholds = [-1.0, 1.0, 100, 1000, Inf]

    all_results = get_all_results(num, get_combined_paths(results_loc))
    all_args = get_all_args(num, get_combined_paths(results_loc))

    block_1 = [3, 4]
    block_2 = [1, 2, 5, 6]

    p = gplot("Time", "Entanglement entropy")
    noentcount = 0

    exacts = exact_states(all_args[1])
    exact_entropies = [entanglement_entropy(state, block_1, block_2) for state in exacts]
    plot!(
        p,
        0:all_args[1]["dt"]:all_args[1]["dt"]*all_args[1]["n_steps"],
        exact_entropies,
        label = "Exact",
        color = "black",
        linestyle = :dash,
        linewidth = 2,
    )


    for (result, args) in zip(all_results, all_args)
        entropies = get_entropies(result, args, block_1, block_2)
        if !args["restrict_entanglement"]
            thresh = Inf
        elseif sum(args["entanglement_args"].gate_indices) == 0
            if noentcount < 1
                noentcount += 1
                thresh = 1.0
            else
                continue
            end
        else
            thresh = args["entanglement_args"].threshold
        end
        println(thresh)
        if !(thresh in thresholds)
            continue
        end
        label, color, style = styles("$(thresh)")
        plot!(
            p,
            0:args["dt"]:args["dt"]*args["n_steps"],
            entropies,
            label = label,
            color = color,
            style = style,
            linewidth = 3,
            legend = :topleft,
        )
    end

    savefig(p, "plots/exp$(num)_entanglement_entropy.svg")
    p
end
