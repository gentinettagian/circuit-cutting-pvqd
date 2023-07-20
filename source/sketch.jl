struct Sketch
    setup::Any
    values::Dict{String,Float64}
    n_spins::Int
end

"Create a sketch from a string."
function sketch(s::String; j1 = 1.0, j2 = 0.25, g = 1.0)
    matrix = []
    for row in split(s, "\n")
        the_row = []
        for col in split(row, " ")
            if length(col) > 0
                push!(the_row, col)
            end
        end
        push!(matrix, the_row)
    end
    return Sketch(matrix, Dict("j" => j1, "J" => j2, "g" => g), count("g", s))
end

"Create a weighted graph from a sketch."
function graph(sketch::Sketch)
    spins = Dict()
    edges = []
    spin_count = 0
    for (i, row) in enumerate(sketch.setup)
        for (j, col) in enumerate(row)
            if col == "g"
                if j % 2 == 0
                    error("Spins have to be placed in odd columns of the sketch.")
                end
                spin_count += 1
                spins[(i, j)] = spin_count
            elseif col == "j"
                if j % 2 == 0
                    # horizontal interaction
                    push!(edges, (((i, j - 1), (i, j + 1)), sketch.values["j"]))
                else
                    # vertical interaction
                    push!(edges, (((i - 1, j), (i + 1, j)), sketch.values["j"]))
                end

            elseif col == "J"
                if j % 2 == 0
                    # horizontal interaction
                    push!(edges, (((i, j - 1), (i, j + 1)), sketch.values["J"]))
                else
                    # vertical interaction
                    push!(edges, (((i - 1, j), (i + 1, j)), sketch.values["J"]))
                end
            end
        end
    end
    numbered_edges = []
    for edge in edges
        push!(numbered_edges, ((spins[edge[1][1]], spins[edge[1][2]]), edge[2]))
    end
    return values(spins), numbered_edges
end

