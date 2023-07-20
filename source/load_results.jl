using Plots
using JLD2
using Base.Threads

include("pvqd.jl")

# This file contains methods to load and display results
# saved from previous runs.


begin
    results_loc = "results"
    paths = readdir(results_loc)

    combined_paths = Dict()

    for path in paths
        if occursin("plots", path) | occursin("DS_Store", path)
            continue
        end
        main_path = path[10:12]
        similar_paths = get(combined_paths, main_path, [])
        push!(similar_paths, "$results_loc/$path")
        combined_paths[main_path] = similar_paths
    end

end

"Load and bundle paths for each experiment."
function get_combined_paths(results_loc)
    paths = readdir(results_loc)

    combined_paths = Dict()

    for path in paths
        if occursin("plots", path) | occursin("DS_Store", path)
            continue
        end
        main_path = path[1:12]
        similar_paths = get(combined_paths, main_path, [])
        push!(similar_paths, "$results_loc/$path")
        combined_paths[main_path] = similar_paths
    end
    return combined_paths
end

"Get all paths for a given experiment."
function get_paths_for_nums(nums, results_loc)
    combined_paths = get_combined_paths(results_loc)
    paths = Dict()
    for num in nums
        for (key, val) in combined_paths
            if occursin("_$(num)", key)
                paths[num] = val
                break
            end
        end
    end
    return paths
end

"Make sure all paths are valid."
function load_args_and_catch(path::String)
    try
        args = get_options(path)
        return args
    catch
        println("Error in $path.")
        return nothing
    end
end

"Filter out invalid paths."
function actual_paths(paths)
    actual_paths = []
    for path in paths
        args = load_args_and_catch(path)
        if !isnothing(args)
            push!(actual_paths, path)
        end
    end
    return actual_paths
end


"Load setup args for a given experiment."
function get_args(num, experiment = 1)
    lookat_path = ""
    for (key, value) in combined_paths
        if occursin("$(num)", key)
            lookat_path = value[experiment]
            #println(lookat_path)
            break
        end
    end
    args = load_object(lookat_path * "/setup_args.txt")
    return args
end

"Load all results for a given experiment."
function get_all_results(num, combined_paths)
    paths = []
    for (key, value) in combined_paths
        if occursin("$(num)", key)
            paths = vcat(paths, value)

        end
    end
    paths = actual_paths(paths)
    results = [load_object(path * "/result") for path in paths]
    return results
end

"Load all setup args for a given experiment."
function get_all_args(num, combined_paths)
    paths = []
    for (key, value) in combined_paths
        if occursin("_$(num)", key)
            paths = vcat(paths, value)


        end
    end
    paths = actual_paths(paths)
    results = [load_object(path * "/setup_args.txt") for path in paths]
    return results
end


"Compare two experiments."
function compare(num1, num2)
    args1 = get_args(num1)
    args2 = get_args(num2)
    compare(args1, args2)
end

"Compare two experiments."
function compare(args1::Dict, args2::Dict)
    println("Key | Experiment 1 | Experiment 2")

    all_keys = union(keys(args1), keys(args2))
    for key in all_keys
        val1 = get(args1, key, "Not found")
        val2 = get(args2, key, "Not found")
        color = :black
        if val1 != val2
            color = :red
        end
        printstyled("$key | $val1 | $val2 \n", color = color)
    end
end


