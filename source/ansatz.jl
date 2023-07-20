using Yao, YaoPlots, YaoExtensions
include("sketch.jl")
include("utils.jl")




"Implement hardware-efficient Ansatz from PVQD paper."
function hweff_ansatz(n, d, θ = nothing; rotations = :xy, periodic = false)
    circ = chain(n)
    p_count = 0
    if isnothing(θ)
        θ = zeros(3000)
    end

    if rotations == :xy
        rot1 = Rx
        rot2 = Ry
    elseif rotations == :xx
        rot1 = Rx
        rot2 = Rx
    else
        error("Invalid rotation type.")
    end

    for layer = 1:d
        # add single qubit rotations
        if mod(layer, 2) == 1
            push!(circ, chain(n, [put(i => rot1(θ[i+p_count])) for i = 1:n]))
        else
            push!(circ, chain(n, [put(i => rot2(θ[i+p_count])) for i = 1:n]))
        end
        p_count += n
        # add entangling gates
        circ = push!(circ, chain(n, [Rzz(n, i, i + 1, θ[i+p_count]) for i = 1:n-1]))
        p_count += n - 1
        if periodic
            circ = push!(circ, Rzz(n, n, 1, θ[p_count]))
            p_count += 1
        end
    end
    # add final single qubit rotations
    if mod(d, 2) == 0
        push!(circ, chain(n, [put(i => rot1(θ[i+p_count])) for i = 1:n]))
    else
        push!(circ, chain(n, [put(i => rot2(θ[i+p_count])) for i = 1:n]))
    end

    return circ
end

"Implement block Ansatz for circuit cutting."
function block_ansatz(
    n_spins,
    n_blocks,
    d,
    cross_gates,
    θ = nothing;
    return_param_types = false,
    rotations = :xy,
)
    n = n_spins * n_blocks
    circ = chain(n)
    p_count = 0
    param_types = []
    if isnothing(θ)
        θ = zeros(3000)
    end

    if rotations == :xy
        rot1 = Rx
        rot2 = Ry
    elseif rotations == :xx
        rot1 = Rx
        rot2 = Rx
    else
        error("Invalid rotation type.")
    end

    cross_indices = []
    for layer = 1:d
        # Single block gates
        for block = 1:n_blocks
            if mod(layer, 2) == 1
                push!(
                    circ,
                    chain(
                        n,
                        [
                            put((block - 1) * n_spins + i => rot1(θ[i+p_count])) for
                            i = 1:n_spins
                        ],
                    ),
                )
                for i = 1:n_spins
                    push!(param_types, :Rx)
                end
            else
                push!(
                    circ,
                    chain(
                        n,
                        [
                            put((block - 1) * n_spins + i => rot2(θ[i+p_count])) for
                            i = 1:n_spins
                        ],
                    ),
                )
                for i = 1:n_spins
                    push!(param_types, :Ry)
                end
            end
            p_count += n_spins
            push!(
                circ,
                chain(
                    n,
                    [
                        Rzz(
                            n,
                            (block - 1) * n_spins + i,
                            (block - 1) * n_spins + i + 1,
                            θ[i+p_count],
                        ) for i = 1:n_spins-1
                    ],
                ),
            )
            for i = 1:n_spins-1
                push!(param_types, :Rzz)
            end
            p_count += n_spins - 1
        end
        # Entangling gates
        if cross_gates[layer]
            push!(
                circ,
                chain(
                    n,
                    [
                        Rzz(n, block * n_spins, block * n_spins + 1, θ[p_count+block]) for
                        block = 1:n_blocks-1
                    ],
                ),
            )

            for block = 1:n_blocks-1
                push!(cross_indices, p_count + block)
                push!(param_types, :RzzEnt)
            end
            p_count += n_blocks - 1
        end
    end
    # Final layer of single qubit rotations
    for block = 1:n_blocks
        if mod(d, 2) == 0
            push!(
                circ,
                chain(
                    n,
                    [
                        put((block - 1) * n_spins + i => rot1(θ[i+p_count])) for
                        i = 1:n_spins
                    ],
                ),
            )
            for i = 1:n_spins
                push!(param_types, :Rx)
            end
        else
            push!(
                circ,
                chain(
                    n,
                    [
                        put((block - 1) * n_spins + i => rot2(θ[i+p_count])) for
                        i = 1:n_spins
                    ],
                ),
            )
            for i = 1:n_spins
                push!(param_types, :Ry)
            end
        end
        p_count += n_spins
    end
    if return_param_types
        return circ, cross_indices, param_types
    end
    return circ, cross_indices
end

"Implement block ansatz from sketch."
function sketch_block_ansatz(
    sketch::Sketch,
    d,
    cross_gates,
    θ = nothing;
    rotations = :xy,
    return_param_types = false,
)
    spins, edges = graph(sketch)
    return graph_ansatz(
        spins,
        edges,
        sketch.values,
        d,
        cross_gates,
        θ,
        rotations = rotations,
        return_param_types = return_param_types,
    )

end

"Get Ansatz from a graph."
function graph_ansatz(
    spins,
    edges,
    values,
    d,
    cross_gates,
    θ;
    rotations = :xx,
    return_param_types = false,
)
    n = length(spins)
    circ = chain(n)
    p_count = 0

    if isnothing(θ)
        θ = zeros(3000)
    end

    param_types = []

    if rotations == :xy
        rot1 = Rx
        rot2 = Ry
    elseif rotations == :xx
        rot1 = Rx
        rot2 = Rx
    elseif rotations == :xz
        rot1 = Rx
        rot2 = Rz
    elseif rotations == :full
        println("Full rotations")
    else
        error("Invalid rotation type.")
    end

    cross_indices = []
    for layer = 1:d
        # Single block gates
        if rotations == :full
            push!(circ, chain(n, [put(i => Rx(θ[i+p_count])) for i = 1:n]))
            p_count += n
            push!(circ, chain(n, [put(i => Rz(θ[i+p_count])) for i = 1:n]))
            p_count += n
            push!(circ, chain(n, [put(i => Ry(θ[i+p_count])) for i = 1:n]))
            p_count += n
        elseif mod(layer, 2) == 1
            push!(circ, chain(n, [put(i => rot1(θ[i+p_count])) for i = 1:n]))
            for i in spins
                push!(param_types, :Rx)
            end
            p_count += n
        else
            push!(circ, chain(n, [put(i => rot2(θ[i+p_count])) for i = 1:n]))
            for i in spins
                push!(param_types, :Ry)
            end
            p_count += n
        end


        # Entangling gates
        for edge in edges
            if edge[2] == values["J"]
                # expensive gate
                if cross_gates[layer]
                    p_count += 1
                    push!(circ, Rzz(n, edge[1][1], edge[1][2], θ[p_count]))
                    push!(cross_indices, p_count)
                    push!(param_types, :RzzEnt)
                end
            else
                # normal gate
                p_count += 1
                push!(circ, Rzz(n, edge[1][1], edge[1][2], θ[p_count]))
                push!(param_types, :Rzz)
            end
        end
    end
    # Final layer of single qubit rotations
    if rotations == :full
        push!(circ, chain(n, [put(i => Rx(θ[i+p_count])) for i = 1:n]))
        p_count += n
        push!(circ, chain(n, [put(i => Rz(θ[i+p_count])) for i = 1:n]))
        p_count += n
        push!(circ, chain(n, [put(i => Ry(θ[i+p_count])) for i = 1:n]))
        p_count += n
    elseif mod(d + 1, 2) == 1
        push!(circ, chain(n, [put(i => rot1(θ[i+p_count])) for i = 1:n]))
        for i in spins
            push!(param_types, :Rx)
        end
        p_count += n
    else
        push!(circ, chain(n, [put(i => rot2(θ[i+p_count])) for i = 1:n]))
        for i in spins
            push!(param_types, :Ry)
        end
        p_count += n
    end


    if return_param_types
        return circ, cross_indices, param_types
    end

    return circ, cross_indices
end

### Helper to load ansatz from options
"Get Ansatz from options."
function get_ansatz(options::Dict; return_param_types = false)
    # check if 2d
    if "sketch" in keys(options)
        if options["restrict_entanglement"]
            gate_indices = options["entanglement_args"].gate_indices
        else
            gate_indices = [true for _ = 1:options["d"]]
        end
        if return_param_types
            ansatz, _, param_types = sketch_block_ansatz(
                options["sketch"],
                options["d"],
                gate_indices,
                options["initial_params"],
                rotations = get(options, "rotations", :xy),
                return_param_types = return_param_types,
            )
            return ansatz, param_types
        else
            return sketch_block_ansatz(
                options["sketch"],
                options["d"],
                gate_indices,
                options["initial_params"],
                rotations = get(options, "rotations", :xy),
                return_param_types = return_param_types,
            )[1]
        end
    elseif options["restrict_entanglement"]
        if return_param_types
            ansatz, _, param_types = block_ansatz(
                options["n_spins"],
                options["n_blocks"],
                options["d"],
                options["entanglement_args"].gate_indices,
                options["initial_params"],
                rotations = get(options, "rotations", :xy),
                return_param_types = return_param_types,
            )
            return ansatz, param_types
        else
            return block_ansatz(
                options["n_spins"],
                options["n_blocks"],
                options["d"],
                options["entanglement_args"].gate_indices,
                options["initial_params"],
                rotations = get(options, "rotations", :xy),
                return_param_types = return_param_types,
            )[1]
        end
    end
    return hweff_ansatz(
        options["n_spins"] * options["n_blocks"],
        options["d"],
        options["initial_params"],
        rotations = get(options, "rotations", :xy),
    )

end

###################################################################################################
# TROTTER

"Implement an Ising Trotter step."
function ising_trotter_step(n, dt, J, g)
    # Rx layer
    circ = chain(n, [put(i => Rx(2 * g * dt)) for i = 1:n])
    # Rzz layer
    push!(circ, chain(n, [Rzz(n, i, i + 1, 2 * J * dt) for i = 1:n-1]))
    return circ
end

"Implement block Ising Trotter step."
function block_ising_trotter_step(n_spins, n_blocks, dt, J1, J2, g)
    n = n_spins * n_blocks
    # Rx layer
    circ = chain(n, [put(i => Rx(2 * g * dt)) for i = 1:n])
    for block = 1:n_blocks
        # Rzz layer
        push!(
            circ,
            chain(
                n,
                [
                    Rzz(
                        n,
                        (block - 1) * n_spins + i,
                        (block - 1) * n_spins + i + 1,
                        2 * J1 * dt,
                    ) for i = 1:n_spins-1
                ],
            ),
        )
    end
    # Entangling layer
    push!(
        circ,
        chain(
            n,
            [
                Rzz(n, block * n_spins, block * n_spins + 1, 2 * J2 * dt) for
                block = 1:n_blocks-1
            ],
        ),
    )
    return circ
end


"Implement Trotter step for 2D-sketch."
function sketch_trotter(sketch::Sketch, dt)
    spins, edges = graph(sketch)
    return graph_trotter(spins, edges, sketch.values["g"], dt)
end


"Implement Trotter step for a graph"
function graph_trotter(spins, edges, g, dt)
    n = length(spins)
    # Rx layer
    circ = chain(n, [put(i => Rx(2 * g * dt)) for i = 1:n])
    # Rzz layer
    push!(circ, chain(n, [Rzz(n, k, l, 2 * j * dt) for ((k, l), j) in edges]))
    return circ
end