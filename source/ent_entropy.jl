
include("load_results.jl")
include("plot.jl")
include("utils.jl")

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

"Get the time at which the threshold saturates."
function get_saturation(result::PVQDResult, times, thresh)
    overheads = result.overheads
    for (i, overhead) in enumerate(overheads)
        if abs(thresh - overhead) / thresh < 0.001
            return times[i]
        end
    end
end

"Plot parameter evolution."
function plot_params(result, args; legend = false)
    _, param_types = get_ansatz(args, return_param_types = true)
    dt = args["dt"]
    n_steps = args["n_steps"]
    color_dict = Dict(
        :Rx => color_scheme[1],
        :Ry => color_scheme[1],
        :Rzz => color_scheme[2],
        :RzzEnt => :red,
    )
    linewidth_dict = Dict(:Rx => 2, :Ry => 2, :Rzz => 2, :RzzEnt => 3)
    params = result.params
    t = range(0, stop = dt * n_steps, length = n_steps + 1)
    plot = gplot("Time", "Parameter value", ratio = 1 / 1.2)
    for i = 1:length(params[1])
        Plots.plot!(
            plot,
            t,
            [p[i] for p in params],
            label = nothing,
            color = color_dict[param_types[i]],
            linewidth = linewidth_dict[param_types[i]],
        )
    end
    saturation = get_saturation(
        result,
        0:args["dt"]:args["dt"]*args["n_steps"],
        args["entanglement_args"].threshold,
    )
    if args["entanglement_args"].threshold > 1 && args["entanglement_args"].threshold < Inf
        vline!(plot, [saturation], label = "", color = :black, linestyle = :dash)
    end
    if legend != false
        plot!(
            plot,
            [],
            [],
            linewidth = 2,
            color = color_scheme[1],
            label = "Single-qubit gates",
        )
        plot!(
            plot,
            [],
            [],
            linewidth = 2,
            color = color_scheme[2],
            label = "Two-qubit gates (within block)",
        )
        plot!(
            plot,
            [],
            [],
            linewidth = 3,
            color = :red,
            label = "Two-qubit gates (between blocks)",
            legend = legend,
        )
    end
    return plot
end


begin
    num = "008"
    results_loc = "results"
    thresholds = [-1.0, 1.0, 100, 1000, Inf]

    all_results = get_all_results(num, get_combined_paths(results_loc))
    all_args = get_all_args(num, get_combined_paths(results_loc))

    actual_τ = [
        sum(args["entanglement_args"].gate_indices) != 0 ?
        args["entanglement_args"].threshold : (args["cut_trotter"] ? 1.0 : -1.0) for
        args in all_args
    ]
    order = reverse(sortperm(actual_τ))
    all_results = all_results[order]
    all_args = all_args[order]
    actual_τ = actual_τ[order]

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
        if thresh == 100.0 || thresh == 1000.0
            saturation =
                get_saturation(result, 0:args["dt"]:args["dt"]*args["n_steps"], thresh)
            vline!(p, [saturation], label = "", color = color, linestyle = :dash)
        end
    end

    bpa_lab, bpa_col, bpa_sty = styles("-1.0")

    Plots.plot!(p, [], [], label = bpa_lab, color = bpa_col, style = bpa_sty, linewidth = 3)

    savefig(p, "plots/exp$(num)_entanglement_entropy.svg")
    p
end

begin
    num = "008"
    τ = 1000
    index = actual_τ .== τ
    res = all_results[index][1]
    args = all_args[index][1]
    p2 = plot_params(res, args, legend = false)
    title!(p2, "τ = 1000")
    ylims!(p2, -2, 2)
    sfont = font(18, "Computer Modern")
    ffont = font(16, "Computer Modern")
    plot!(
        p2,
        legendfont = sfont,
        tickfont = ffont,
        xlabelfont = sfont,
        ylabelfont = sfont,
        guidefont = sfont,
        titlefont = sfont,
    )

    savefig(p2, "plots/exp$(num)_$(τ)_params.svg")

    legend_plot = Plots.plot(
        legendfont = sfont,
        tickfont = ffont,
        xlabelfont = sfont,
        ylabelfont = sfont,
        guidefont = sfont,
        titlefont = sfont,
    )
    Plots.plot!(
        legend_plot,
        [],
        [],
        linewidth = 1,
        color = color_scheme[1],
        label = "Single-qubit gates",
    )
    Plots.plot!(
        legend_plot,
        [],
        [],
        linewidth = 1,
        color = color_scheme[2],
        label = "Two-qubit gates (within block)",
    )
    Plots.plot!(
        legend_plot,
        [],
        [],
        linewidth = 2,
        color = :red,
        label = "Two-qubit gates (between blocks)",
    )
    savefig(legend_plot, "plots/params_legend.svg")


end