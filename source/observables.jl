using Plots
using LinearAlgebra
using Yao, YaoExtensions
using Measures

include("hamiltonians.jl")
include("ansatz.jl")
include("utils.jl")
include("sketch.jl")

"Create exact states for the tfim model."
function exact_states(n_spins, n_blocks, J1, J2, g, dt, n_steps, initial_state)
    ham = block_tfim_ham(n_spins, n_blocks, J1, J2, g)
    d, v = eigen(ham)
    evolved(t1) = v * diagm(exp.(-1im * t1 * d)) * v' * initial_state

    times = range(0, stop = dt * n_steps, length = n_steps + 1)
    return [evolved(t) for t in times]
end

"Create exact states from options."
function exact_states(options)
    if haskey(options, "sketch")
        return exact_states(options["sketch"], options)
    end
    n_spins = options["n_spins"]
    n_blocks = options["n_blocks"]
    J1 = options["J1"]
    J2 = options["J2"]
    g = options["g"]
    dt = options["dt"]
    n_steps = options["n_steps"]
    initial_state = Yao.state(zero_state(n_spins * n_blocks))
    return exact_states(n_spins, n_blocks, J1, J2, g, dt, n_steps, initial_state)
end

"Create exact states from sketch."
function exact_states(sketch::Sketch, options)
    ham = hamiltonian(sketch)
    return exact_states(ham, options)
end

"Create exact states from hamiltonian."
function exact_states(ham, options)
    initial_state = zeros(Complex, size(ham)[1])
    initial_state[1] = 1.0
    d, v = eigen(ham)
    evolved(t1) = v * diagm(exp.(-1im * t1 * d)) * v' * initial_state
    dt = options["dt"]
    n_steps = options["n_steps"]
    times = range(0, stop = dt * n_steps, length = n_steps + 1)
    return [evolved(t) for t in times]
end


"Generate states from the PVQD results."
function pvqd_states(result, ansatz)
    states = []
    for params in result.params
        dispatch!(ansatz, params)
        state = Yao.state(zero_state(ansatz.n) |> ansatz)
        push!(states, state)
    end
    return states
end

"Measure x-magnetization."
function x_magnetization(n, states)
    obs = applymatrix(sum(put(n, i => X) for i = 1:n)) / n
    return observe(obs, states)
end

"Measure x-magnetization."
function x_magnetization(states)
    n = Int(log2(length(states[1])))
    obs = applymatrix(sum(put(n, i => X) for i = 1:n)) / n
    return observe(obs, states)
end


"Measure z-magnetization."
function z_magnetization(n, states)
    obs = applymatrix(sum(put(n, i => Z) for i = 1:n)) / n
    return observe(obs, states)
end

"Measure z-magnetization."
function z_magnetization(states)
    n = Int(log2(length(states[1])))
    obs = applymatrix(sum(put(n, i => Z) for i = 1:n)) / n
    return observe(obs, states)
end

"Measure any observable."
function observe(obs, states)
    return [real.(state' * obs * state)[1] for state in states]
end
