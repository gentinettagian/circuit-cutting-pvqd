include("load_results.jl")
include("utils.jl")
include("plot.jl")
using Plots


"Calculate total number of shots and errors for each experiment."
function get_sample_data(experiments, result_loc = "results")
    comb_paths = get_combined_paths(result_loc)
    data = Dict()

    for exp in experiments
        results = get_all_results(exp, comb_paths)
        args = get_all_args(exp, comb_paths)
        solutions = exact_states(args[1])

        for (setup, result) in zip(args, results)
            r_shots = setup["shots"]
            if setup["restrict_entanglement"]
                if sum(setup["entanglement_args"].gate_indices) == 0
                    if setup["cut_trotter"]
                        τ = 1.0
                    else
                        τ = -1.0
                    end
                else
                    τ = setup["entanglement_args"].threshold
                end
            else
                τ = Inf
            end
            τ_data = get(data, τ, [])

            ansatz = get_ansatz(setup)
            n_params = length(parameters(ansatz))
            overheads = result.overheads
            t_shots = sum(r_shots .* overheads) * setup["opt_args"].maxiter * 2 * n_params
            states = pvqd_states(result, get_ansatz(setup))
            #error = 1 - fidelity(states[end], solutions[end])
            error = mean([
                1 - fidelity(state, solution) for
                (state, solution) in zip(states, solutions)
            ])
            push!(τ_data, [r_shots, t_shots, error])
            data[τ] = τ_data
        end
    end
    for (τ, τ_data) in data
        data[τ] = reduce(vcat, transpose.(τ_data))
    end
    return data
end

function plot_sample_data(data)
    raw_plot = gplot(L"Raw shots $R$", "Error")
    plot!(raw_plot, xaxis = :log, yaxis = :log, legend = false)

    tot_plot = gplot(L"Total shots $R_{tot}$", "Mean infidelity")
    plot!(tot_plot, xaxis = :log, yaxis = :identity, legend = false)

    all_plot = gplot("Total shots", "Error")
    plot!(all_plot, xaxis = :log, yaxis = :log, legend = false)

    taus = sort(collect(keys(data)))

    for τ in taus
        if τ == 500
            continue
        end
        τ_data = data[τ]
        label, color, style, marker = styles("$(τ)", return_marker = true)

        unique_shots = unique(τ_data[:, 1])
        means = zeros((length(unique_shots), 3))
        stds = zeros((length(unique_shots), 3))

        for (i, shots) in enumerate(unique_shots)
            shots_data = τ_data[τ_data[:, 1].==shots, :]
            means[i, :] = mean(shots_data, dims = 1)
            stds[i, :] = std(shots_data, dims = 1)
        end
        println(means)

        scatter!(
            raw_plot,
            means[:, 1],
            means[:, 3],
            label = label,
            yerror = stds[:, 3],
            ms = 5,
            marker = marker,
            color = color,
        )
        if τ == Inf
            continue
        end
        scatter!(
            tot_plot,
            means[:, 2],
            means[:, 3],
            label = label,
            yerror = stds[:, 3],
            xerror = stds[:, 2],
            ms = 5,
            marker = marker,
            color = color,
            legend = false,
        )

        scatter!(all_plot, τ_data[:, 2], τ_data[:, 3], label = label, ms = 7, color = color)
    end

    return raw_plot, tot_plot, all_plot
end

function plot_sv_fidelity!(plots, exp_num, result_loc = "results")
    comb_paths = get_combined_paths(result_loc)
    results = get_all_results(exp_num, comb_paths)
    args = get_all_args(exp_num, comb_paths)
    solutions = exact_states(args[1])

    for (setup, result) in zip(args, results)
        ansatz = get_ansatz(setup)
        states = pvqd_states(result, ansatz)
        τ = setup["entanglement_args"].threshold

        if !setup["cut_trotter"]
            τ = -1.0
        elseif sum(setup["entanglement_args"].gate_indices) == 0
            τ = 1.0
        end
        if !(τ in [-1, 1, 100, 1000])
            continue
        end

        label, color, style = styles("$(τ)")
        fid = mean([
            1 .- fidelity(state, solution) for (state, solution) in zip(states, solutions)
        ])



        for plot in plots
            Plots.hline!(plot, [fid], label = nothing, color = color, style = :dash)
        end
    end
end



experiments = [900, 901, 903, 904, 906, 907]
data = get_sample_data(experiments, "results")

plots = plot_sample_data(data)

plot_sv_fidelity!(plots, 805)
ylims!(plots[2], (0.0, 0.075))
plots[2]
savefig(plots[2], "plots/900-907_total_shots.svg")