using Yao, YaoExtensions, Yao.AD
import Optimisers
using Zygote
using Statistics


include("ansatz.jl")
include("fidelity.jl")
include("gradients.jl")
include("optimization.jl")
include("sketch.jl")




struct PVQDResult
    params::Array{Array{Float64,1},1}
    initial_fidelities::Array{Float64,1}
    fidelities::Array{Float64,1}
    success::Array{Symbol,1}
    overheads::Array{Float64,1}
    iterations::Array{Int64,1}
    histories::Array{Array{Float64,2},1}
end

struct EntanglementArgs
    gate_indices::Array{Bool,1}
    threshold::Float64
    backstep_size::Float64
    reset_momentum::Bool
    reset_v::Bool
end

struct OptimizerArgs
    lr::Float64
    abstol::Union{Float64,Nothing}
    gradtol::Union{Float64,Nothing}
    rel_ftol::Union{Float64,Nothing}
    rel_gtol::Union{Float64,Nothing}
    bin_size::Union{Int64,Nothing}
    maxiter::Int64
    method::Symbol
    other::Any
    gradient_method::Symbol
    shift::Float64 # for parameter shift
end


struct PVQD
    ansatz::ChainBlock
    n_steps::Int
    initial_params::Vector{Float64}
    trotter_step::ChainBlock
    optimizer_args::OptimizerArgs
    restrict_entanglement::Bool
    entanglement_args::Union{EntanglementArgs,Nothing}
    entanglement_indices::Union{Array{Int,1},Nothing}
    loss::Symbol
    n_shots::Union{Int64,Vector{Int64},Nothing}
    guess_strategy::Symbol
    initial_delta::Float64
    cut_overhead::Float64
end

"Construct PVQD object for Ising Chain of n_blocks with n_spins each."
function PVQD(
    n_spins,
    n_blocks,
    depth,
    rotations,
    J1,
    J2,
    g,
    n_steps,
    dt,
    initial_params,
    optimizer_args,
    restrict_entanglement,
    entanglement_args,
    loss,
    n_shots,
    guess_strategy = :delta,
    initial_delta = 0.0,
    cut_trotter = true,
)

    if !cut_trotter
        trotter_step = block_ising_trotter_step(n_spins, n_blocks, dt, J1, 0, g)
        cut_overhead = 1.0
    else
        trotter_step = block_ising_trotter_step(n_spins, n_blocks, dt, J1, J2, g)
        # sqrt because trotter step is not doubled
        cut_overhead = sqrt(gamma(2 * J2 * dt))^(n_blocks - 1)
    end

    if restrict_entanglement
        ansatz, indices = block_ansatz(
            n_spins,
            n_blocks,
            depth,
            entanglement_args.gate_indices,
            initial_params,
            rotations = rotations,
        )
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
    else
        ansatz =
            hweff_ansatz(n_spins * n_blocks, depth, initial_params, rotations = rotations)
        return PVQD(
            ansatz,
            n_steps,
            initial_params,
            trotter_step,
            optimizer_args,
            restrict_entanglement,
            entanglement_args,
            nothing,
            loss,
            n_shots,
            guess_strategy,
            initial_delta,
            cut_overhead,
        )
    end

end


"Construct PVQD object for 2d sketch."
function PVQD(
    sketch::Sketch;
    depth,
    rotations,
    n_steps,
    dt,
    initial_params,
    optimizer_args,
    restrict_entanglement,
    entanglement_args,
    loss,
    n_shots,
    guess_strategy = :delta,
    initial_delta = 0.0,
    cut_trotter = true,
)

    if cut_trotter
        trotter_step = sketch_trotter(sketch, dt)
        cross_edges = 0
        for bb in sketch.setup
            for b in bb
                if b == "J"
                    cross_edges += 1
                end
            end
        end
        cut_overhead = sqrt(gamma(2 * sketch.values["J"] * dt))^cross_edges

    else
        # For BPA, the Trotter step is a block product
        new_sketch = Sketch(
            sketch.setup,
            Dict("j" => sketch.values["j"], "g" => sketch.values["g"], "J" => 0.0),
            sketch.n_spins,
        )
        trotter_step = sketch_trotter(new_sketch, dt)
        cut_overhead = 1.0
    end

    if restrict_entanglement
        ansatz, indices = sketch_block_ansatz(
            sketch,
            depth,
            entanglement_args.gate_indices,
            initial_params,
            rotations = rotations,
        )
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
    else
        ansatz, indices = sketch_block_ansatz(
            sketch,
            depth,
            [true for _ = 1:depth],
            initial_params,
            rotations = rotations,
        )
        return PVQD(
            ansatz,
            n_steps,
            initial_params,
            trotter_step,
            optimizer_args,
            restrict_entanglement,
            entanglement_args,
            nothing,
            loss,
            n_shots,
            guess_strategy,
            initial_delta,
            cut_overhead,
        )
    end

end

"Define the standard PVQD loss and gradient."
function pvqd_loss_and_grad(compute_uncompute, observable, n_shots, gradient_method, shift)
    if isnothing(n_shots)
        fid = real(expect(observable, zero_state(compute_uncompute.n) |> compute_uncompute))
        if gradient_method == :exact
            grad =
                -expect'(observable, zero_state(compute_uncompute.n) => compute_uncompute)[2]
            return 1 - fid, grad
        end
    else
        fid = sample(compute_uncompute, observable, n_shots)
    end
    if gradient_method == :param_shift
        grad = .-param_shift(compute_uncompute, observable, n_shots; shift = shift)
    elseif gradient_method == :spsa
        grad = .-spsa_grad(compute_uncompute, observable, n_shots)
    else
        error("Only param_shift and spsa are implemented for shot based simulation.")
    end
    return 1 - fid, grad

end




"Penalty for entanglement."
function penalty(theta, pvqd::PVQD)
    g = gamma(theta) * pvqd.cut_overhead
    if g > pvqd.entanglement_args.threshold
        return g
    end
    return 0.0
end

"Penalty gradient."
function penalty_grad(theta, pvqd::PVQD)
    g = gamma(theta) * pvqd.cut_overhead
    if g > pvqd.entanglement_args.threshold
        return gamma'(theta) * pvqd.cut_overhead
    end
    return zeros(length(theta))
end


"If above threshold move back until below."
function move_back(x, pvqd)
    x_new = deepcopy(x)
    pen = theta -> penalty(theta, pvqd)

    steps = 0
    while pen(x_new[pvqd.entanglement_indices]) > 0
        steps += 1
        gradient = penalty_grad(x_new[pvqd.entanglement_indices], pvqd)
        gradient ./= norm(gradient)
        x_new[pvqd.entanglement_indices] -= pvqd.entanglement_args.backstep_size * gradient
    end
    return x_new, steps
end



"Define the entanglement penalty loss and gradient."
function construct_loss(pvqd::PVQD)
    if pvqd.loss == :global
        observable = zero_proj(pvqd.ansatz.n)
    elseif pvqd.loss == :local
        observable = local_proj(pvqd.ansatz.n)
    else
        error("Only global or local loss is implemented.")
    end

    if isnothing(pvqd.n_shots) || isa(pvqd.n_shots, Int)

        loss_and_grad =
            compute_uncompute -> pvqd_loss_and_grad(
                compute_uncompute,
                observable,
                pvqd.n_shots,
                pvqd.optimizer_args.gradient_method,
                pvqd.optimizer_args.shift,
            )
    else
        loss_and_grad =
            (compute_uncompute, shots) -> pvqd_loss_and_grad(
                compute_uncompute,
                observable,
                shots,
                pvqd.optimizer_args.gradient_method,
                pvqd.optimizer_args.shift,
            )

    end
    return loss_and_grad

end



"Perform a single PVQD timestep."
function pvqd_step(pvqd::PVQD, params, delta, loss_and_grad::Function)
    # Bind new parameters to the ansatz
    dispatch!(pvqd.ansatz, params)
    # Define the trotter-evolved state
    trotterized = chain(NoParams(pvqd.trotter_step), NoParams(pvqd.ansatz'))


    # Define the fidelity circuit
    compute_uncompute = chain(pvqd.ansatz, trotterized)
    if pvqd.guess_strategy == :delta
        x = parameters(compute_uncompute) .+ delta
    elseif pvqd.guess_strategy == :previous
        x = parameters(compute_uncompute)
    elseif pvqd.guess_strategy == :zero
        x = zeros(length(parameters(compute_uncompute)))
    elseif pvqd.guess_strategy == :random
        x = rand(length(parameters(compute_uncompute))) * 2 * pi
    else
        error(
            "Invalid guess strategy. Only :delta, :previous, :random and :zero are implemented.",
        )
    end
    dispatch!(compute_uncompute, x)
    optimizer = construct(pvqd.optimizer_args, x)
    loss, g = loss_and_grad(compute_uncompute)
    history = zeros(pvqd.optimizer_args.maxiter, 3)
    initial_fid = 1 - loss
    fid = initial_fid
    success = :maxiter
    count = 0

    for i = 1:pvqd.optimizer_args.maxiter
        count += 1
        if i > 1
            loss, g = loss_and_grad(compute_uncompute)
            fid = 1 - loss
        end
        Optimisers.update!(optimizer, x, g)

        if pvqd.restrict_entanglement
            # we check if the entanglement is too high and move back if needed
            x, backsteps = move_back(x, pvqd)
            history[i, 3] = backsteps

            if pvqd.entanglement_args.reset_momentum && pvqd.optimizer_args.method == :adam
                # reset momentum for enganglement parameters
                optimizer.state[1][pvqd.entanglement_indices] .= 0
            end
            if pvqd.entanglement_args.reset_v && pvqd.optimizer_args.method == :adam
                # reset v for enganglement parameters
                optimizer.state[2][pvqd.entanglement_indices] .= 0
            end

        end

        # update the circuit
        dispatch!(compute_uncompute, x)

        history[i, 1] = loss
        history[i, 2] = maximum(abs.(g))

        termination = terminate(history, i, pvqd.optimizer_args)
        if termination != :continue
            success = termination
            break
        end

    end

    overhead =
        pvqd.restrict_entanglement ?
        gamma(parameters(pvqd.ansatz)[pvqd.entanglement_indices]) : 1.0
    overhead = overhead * pvqd.cut_overhead

    println("Overhead $(overhead)")

    return x, initial_fid, fid, success, overhead, count, history

end



"Run the PVQD algorithm."
function run(pvqd::PVQD)
    params = [pvqd.initial_params]
    initial_fidelities = []
    fidelities = []
    successes = []
    overheads = []
    iterations = []
    histories = []
    loss_and_grad = construct_loss(pvqd)
    delta = rand(length(pvqd.initial_params)) * pvqd.initial_delta
    for i = 1:pvqd.n_steps
        # fix number of shots if depending on step
        if (isnothing(pvqd.n_shots) || isa(pvqd.n_shots, Int))
            l_n_g = loss_and_grad
        else
            l_n_g = compute_uncompute -> loss_and_grad(compute_uncompute, pvqd.n_shots[i])
        end

        new_params, initial_fid, fid, success, overhead, count, history =
            pvqd_step(pvqd, params[i], delta, l_n_g)
        push!(initial_fidelities, initial_fid)
        push!(fidelities, fid)
        push!(successes, success)
        push!(overheads, overhead)
        push!(iterations, count)
        push!(histories, history)
        println("Step $i, fidelity = $(fid), termination: $(success)")
        delta = new_params - params[i]

        push!(params, params[i] .+ delta)
    end
    return PVQDResult(
        params,
        initial_fidelities,
        fidelities,
        successes,
        overheads,
        iterations,
        histories,
    )

end


