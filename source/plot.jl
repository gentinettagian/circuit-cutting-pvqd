
using Plots, LinearAlgebra
using LaTeXStrings
using Yao, JLD2
using Measures

include("ansatz.jl")
include("observables.jl")
include("pvqd.jl")



struct PlotData
    path::String
    states::Any
    result::PVQDResult
    times::Any
end




color_scheme = ["#66C5CC", "#F6CF71", "#F89C74", "#DCB0F2", "#87C55F", "#B3B3B3"]

"Get plotting styles according to the threshold."
function styles(label; return_marker = false)
    if label == "-1.0"
        better_label = "BPA"
        color = color_scheme[1]
        style = :solid
        marker = :star5

    elseif label == "1.0"
        better_label = "BPA*"
        color = color_scheme[2]
        style = :dash
        marker = :hexagon

    elseif label == "1.0e9" || label == "Inf"
        better_label = "CKA, no threshold"
        color = color_scheme[5]
        style = :solid
        marker = :circle
    else
        if label == "100.0"
            better_label = "CKA, τ = $(label[1:end-2])"
            color = color_scheme[3]
            style = :dot
            marker = :diamond
        elseif label == "1000.0"
            better_label = "CKA, τ = $(label[1:end-2])"
            color = color_scheme[4]
            style = :dashdot
            marker = :utriangle
        elseif label == "10.0"
            better_label = "CKA, τ = $(label[1:end-2])"
            color = color_scheme[3]
            style = :dot
            marker = :utriangle
        else
            better_label = label
            color = :black
            style = :solid
            marker = :rectangle
        end
    end

    if return_marker
        return better_label, color, style, marker
    end
    return better_label, color, style

end


"Load the ansatz from options."
function ansatz_from(options)
    if options["restrict_entanglement"]
        ansatz, _ = block_ansatz(
            options["n_spins"],
            options["n_blocks"],
            options["d"],
            options["entanglement_args"].gate_indices,
            options["initial_params"],
            rotations = get(options, "rotations", :xy),
        )
    else
        ansatz = hweff_ansatz(
            options["n_spins"] * options["n_blocks"],
            options["d"],
            options["initial_params"],
            rotations = get(options, "rotations", :xy),
        )
    end

    return ansatz
end

"Prepare data."
function prepare_data(exp_num::String, thresholds; results_loc = "results")
    paths = readdir(results_loc)
    paths = filter(path -> occursin("_$(exp_num)_", path), paths)
    println(paths)
    data = Dict()
    ex_states = []
    for path in paths
        for tau in thresholds
            if occursin("_$(tau)_", path[10:end]) ||
               (tau < Inf && occursin("_$(Int(tau))_", path[10:end]))
                result = nothing
                options = nothing
                try
                    result = load_object("$results_loc/$path/result")
                    options = load_object("$results_loc/$path/setup_args.txt")
                catch
                    println("Experiment $path not found.")
                    continue
                end
                if results_loc == "j1j2_results"
                    ansatz, _ = graph_ansatz(
                        options["spins"],
                        options["edges"],
                        Dict("j" => options["j1"], "J" => options["j2"]),
                        options["depth"],
                        options["ent_args"].gate_indices,
                        options["initial_params"],
                        rotations = options["rotations"],
                    )
                    if length(ex_states) == 0
                        ham = hamiltonian(options["spins"], options["edges"], options["g"])
                        ex_states = exact_states(ham, options)
                    end
                else
                    ansatz = get_ansatz(options)
                    if length(ex_states) == 0
                        ex_states = exact_states(options)
                    end
                end
                states = pvqd_states(result, ansatz)

                plot_data = PlotData(
                    path,
                    states,
                    result,
                    range(
                        0,
                        stop = options["dt"] * options["n_steps"],
                        length = options["n_steps"] + 1,
                    ),
                )
                push!(get!(data, "$tau", []), plot_data)

            end
        end
    end


    return sort(collect(data), by = x -> -parse(Float64, x[1])), ex_states
end

"Plot fidelity."
function plot_fidelity(data, ex_states; options = nothing, infidelity = false)
    p = gplot("Time", "Fidelity", options)
    plot!(p, [0, 2], [1, 1], color = :dimgray, linewidth = 1, label = nothing)

    min_fid = 1.0
    for (label, plot_datas) in data
        fidelities = zeros(length(plot_datas), length(plot_datas[1].states))
        for (index, plot_data) in enumerate(plot_datas)
            for (time_index, state) in enumerate(plot_data.states)
                fidelities[index, time_index] = fidelity(state, ex_states[time_index])
            end
        end
        if infidelity
            fidelities = 1 .- fidelities
        end
        mean_fidelities = mean(fidelities, dims = 1)[1, :]
        lower_error =
            mean_fidelities .-
            [quantile(fidelities[:, i], 0.159) for i = 1:size(fidelities, 2)]
        upper_error =
            [quantile(fidelities[:, i], 0.841) for i = 1:size(fidelities, 2)] .- mean_fidelities
        std_fidelities = std(fidelities, dims = 1)[1, :]
        better_label, color, style = styles(label)
        Plots.plot!(
            p,
            plot_datas[1].times,
            mean_fidelities,
            ribbon = (lower_error, upper_error),
            label = better_label,
            color = color,
            linewidth = 3,
            linestyle = style,
        )
        min_fid = minimum([min_fid, minimum(mean_fidelities)])
        println("$(label): Final fidelity $(mean_fidelities[end]) ± $(std_fidelities[end])")
    end
    min_fid = floor(min_fid * 50) / 50
    yticks!(p, min_fid:0.02:1.0)
    return p
end

"Plot iterations"
function plot_iterations(data; options = nothing)
    p = gplot("Time", "Iterations", options)
    for (label, plot_datas) in data
        iterations = zeros(length(plot_datas), length(plot_datas[1].result.iterations))
        for (index, plot_data) in enumerate(plot_datas)
            iterations[index, :] = plot_data.result.iterations
        end
        mean_iterations = mean(iterations, dims = 1)[1, :]
        lower_error =
            mean_iterations .-
            [quantile(iterations[:, i], 0.159) for i = 1:size(iterations, 2)]
        upper_error =
            [quantile(iterations[:, i], 0.841) for i = 1:size(iterations, 2)] .- mean_iterations
        better_label, color, style = styles(label)
        Plots.plot!(
            p,
            plot_datas[1].times[2:end],
            mean_iterations,
            ribbon = (lower_error, upper_error),
            label = better_label,
            color = color,
            linewidth = 3,
            linestyle = style,
            legend = :topleft,
        )
    end
    return p
end


"Plot any observable."
function plot_observable(observable::Tuple{Any,Any}, data, ex_states; options = nothing)
    name, observable = observable
    p = gplot("Time", name, options)
    plotted_exact = false
    for (label, plot_datas) in data
        if !plotted_exact
            exact_magnetizations = observe(observable, ex_states)
            Plots.plot!(
                p,
                plot_datas[1].times,
                exact_magnetizations,
                label = "Exact",
                color = :black,
                linestyle = :dash,
                linewidth = 2,
            )
            plotted_exact = true
        end

        magnetizations = zeros(length(plot_datas), length(plot_datas[1].states))
        for (index, plot_data) in enumerate(plot_datas)
            magnetizations[index, :] = observe(observable, plot_data.states)
        end
        mean_magnetizations = mean(magnetizations, dims = 1)[1, :]
        lower_error =
            mean_magnetizations .-
            [quantile(magnetizations[:, i], 0.159) for i = 1:size(magnetizations, 2)]
        upper_error =
            [quantile(magnetizations[:, i], 0.841) for i = 1:size(magnetizations, 2)] .- mean_magnetizations
        std_magnetizations = std(magnetizations, dims = 1)[1, :]

        better_label, color, style = styles(label)
        Plots.plot!(
            p,
            plot_datas[1].times,
            mean_magnetizations,
            ribbon = (lower_error, upper_error),
            label = better_label,
            color = color,
            linewidth = 3,
            linestyle = style,
        )
    end

    return p
end

"Plot overhead"
function plot_overhead(data; options = nothing, scale = :log)
    p = gplot("Time", "Overhead", options)

    for (label, plot_datas) in data
        overheads = zeros(length(plot_datas), length(plot_datas[1].result.overheads))
        for (index, plot_data) in enumerate(plot_datas)
            overheads[index, :] = plot_data.result.overheads
        end
        mean_overheads = median(overheads, dims = 1)[1, :]
        lower_error =
            mean_overheads .-
            [quantile(overheads[:, i], 0.159) for i = 1:size(overheads, 2)]
        upper_error =
            [quantile(overheads[:, i], 0.841) for i = 1:size(overheads, 2)] .- mean_overheads
        better_label, color, style = styles(label)
        Plots.plot!(
            p,
            plot_datas[1].times[2:end],
            mean_overheads,
            ribbon = (lower_error, upper_error),
            label = better_label,
            color = color,
            linestyle = style,
            linewidth = 3,
            yaxis = scale,
        )
    end
    yticks!(p, [1, 10, 100, 1000], ["1", "10", "100", "1000"])
    return p
end

"Plot fidelity vs overhead."
function plot_fid_ovhd(data; options = nothing, scale = :log)
    p = gplot("Overhead", "Infidelity", options)
    overhead_points = []
    fid_means = []
    for (label, plot_datas) in data
        # get overheads
        overheads = zeros(length(plot_datas), length(plot_datas[1].result.overheads))
        for (index, plot_data) in enumerate(plot_datas)
            overheads[index, :] = plot_data.result.overheads
        end
        mean_overheads = mean(overheads, dims = 1)[1, :]
        # get fidelities
        fidelities = zeros(length(plot_datas), length(plot_datas[1].states))
        for (index, plot_data) in enumerate(plot_datas)
            for (time_index, state) in enumerate(plot_data.states)
                fidelities[index, time_index] = fidelity(state, ex_states[time_index])
            end
        end
        mean_fidelities = mean(fidelities, dims = 1)[1, :]

        std_fidelities = std(fidelities, dims = 1)[1, :]
        better_label, color, _, marker = styles(label, return_marker = true)
        if marker != :rectangle && label != "10.0"
            Plots.scatter!(
                p,
                [mean_overheads[end]],
                [1 - mean(mean_fidelities)],
                label = better_label,
                color = color,
                marker = marker,
            )
        else
            push!(overhead_points, mean_overheads[end])
            push!(fid_means, mean(mean_fidelities))
        end

        println(mean_overheads[end], " ", mean_fidelities[end])
    end
    Plots.scatter!(
        p,
        overhead_points,
        1 .- fid_means,
        xaxis = scale,
        yaxis = :identity,
        label = false,
        color = :black,
    )
    println(overhead_points)
    return p
end


# Spin chain experiments.
begin
    num = "805" # Choose experiment number
    results_loc = "results" # Directory where results are stored

    # Thresholds to be plotted
    thresholds = [-1.0, 1.0, 100, 1000, Inf]

    data, ex_states = prepare_data(num, thresholds, results_loc = results_loc)

    # Fidelity plot
    p_fid = plot_fidelity(
        data,
        ex_states,
        options = Dict("yaxis" => :left, "legend" => :bottomleft),
    )
    savefig("plots/exp$(num)_fid.svg")

    # Overhead plot
    p_ovhd = plot_overhead(
        data,
        scale = :log,
        options = Dict("yaxis" => :left, "legend" => false),
    )

    savefig("plots/exp$(num)_overhead.svg")

    # Observable
    obs = (
        L"\langle Z_1X_3Z_5\rangle",
        applymatrix(chain(put(6, 1 => Z), put(6, 3 => X), put(6, 5 => Z))),
    )
    p_x = plot_observable(
        obs,
        data,
        ex_states,
        options = Dict("yaxis" => :left, "legend" => false),
    )
    savefig("plots/exp$(num)_$(obs[1][10:end-8]).svg")


    thresholds = [-1.0, 1.0, 100, 1000, 10, 50, 500, 5, 25, 250, 5000]
    data, ex_states = prepare_data(num, thresholds, results_loc = results_loc)

    p_fo = plot_fid_ovhd(
        data,
        scale = :log10,
        options = Dict("yaxis" => :left, "legend" => :topright),
    )

    ylims!(p_fo, (0.0, 0.075))
    xticks!(p_fo, [1, 1e1, 1e2, 1e3])

    savefig("plots/exp$(num)_fo.svg")
end

# Two-leg ladder experiment
begin
    num = "079" # Choose experiment number
    results_loc = "2dresults" # Directory where results are stored

    # Thresholds to be plotted
    thresholds = [-1.0, 1.0, 100, 1000, Inf]

    data, ex_states = prepare_data(num, thresholds, results_loc = results_loc)

    # Fidelity plot
    p_fid = plot_fidelity(
        data,
        ex_states,
        options = Dict("yaxis" => :left, "legend" => :bottomleft),
    )
    savefig("plots/exp$(num)_fid.svg")

    # Overhead plot
    p_ovhd = plot_overhead(
        data,
        scale = :log,
        options = Dict("yaxis" => :left, "legend" => false),
    )

    savefig("plots/exp$(num)_overhead.svg")

    # Observable
    obs = (
        L"\langle X_1X_4X_5X_8\rangle",
        applymatrix(chain(put(8, 1 => X), put(8, 4 => X), put(8, 5 => X), put(8, 8 => X))),
    )
    p_x = plot_observable(
        obs,
        data,
        ex_states,
        options = Dict("yaxis" => :left, "legend" => false),
    )
    savefig("plots/exp$(num)_$(obs[1][10:end-8]).svg")
end


# J1-J2 experiment
begin
    num = "100" # Choose experiment number
    results_loc = "j1j2_results" # Directory where results are stored

    # Thresholds to be plotted
    thresholds = [-1.0, 1.0, 10, Inf]

    data, ex_states = prepare_data(num, thresholds, results_loc = results_loc)

    # Fidelity plot
    p_fid = plot_fidelity(
        data,
        ex_states,
        options = Dict("yaxis" => :left, "legend" => :bottomleft),
    )
    savefig("plots/exp$(num)_fid.svg")

    # Overhead plot
    p_ovhd = plot_overhead(
        data,
        scale = :log,
        options = Dict("yaxis" => :left, "legend" => false),
    )

    savefig("plots/exp$(num)_overhead.svg")

    # Observable
    obs = (L"\langle X_1Z_3\rangle", applymatrix(chain(put(4, 1 => X), put(4, 3 => Z))))
    p_x = plot_observable(
        obs,
        data,
        ex_states,
        options = Dict("yaxis" => :left, "legend" => false),
    )
    p_x = plot_observable(
        obs,
        data,
        ex_states,
        options = Dict("yaxis" => :left, "legend" => false),
    )
    savefig("plots/exp$(num)_$(obs[1][10:end-8]).svg")

end
