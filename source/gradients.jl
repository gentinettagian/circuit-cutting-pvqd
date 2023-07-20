using Yao
using Distributions
import StatsBase.sample
using Base.Threads

"Sample measurement of a circuit."
function sample(circuit, observable, n_shots)
    return real(
        sum(measure(observable, zero_state(circuit.n) |> circuit, nshots = n_shots)) /
        n_shots,
    )
end

"Implement parameter shift rule to calculate gradients."
function param_shift(circuit, observable, n_shots; shift = pi / 2)
    # Get number of parameters
    params = parameters(circuit)
    n_params = length(params)
    grad = zeros(n_params)
    # Loop over parameters
    for i = 1:n_params
        params_plus = copy(params)

        params_plus[i] += shift
        dispatch!(circuit, params_plus)
        if isnothing(n_shots)
            fid_plus = real(expect(observable, zero_state(circuit.n) |> circuit))
        else
            fid_plus = sample(circuit, observable, n_shots)
        end

        params_minus = copy(params)
        params_minus[i] -= shift
        dispatch!(circuit, params_minus)
        if isnothing(n_shots)
            fid_minus = real(expect(observable, zero_state(circuit.n) |> circuit))
        else
            fid_minus = sample(circuit, observable, n_shots)
        end



        grad[i] = (fid_plus - fid_minus) / (2 * sin(shift))
        dispatch!(circuit, params)

    end
    # Reset circuit to original parameters.

    return grad
end

"Implement SPSA gradient."
function spsa_grad(circuit, observable, n_shots; c = 0.005)
    # Get number of parameters
    params = parameters(circuit)
    n_params = length(params)

    delta = rand(Bernoulli(0.5), n_params) .* 2 .- 1
    params_plus = copy(params) .+ c .* delta
    dispatch!(circuit, params_plus)
    fid_plus = sample(circuit, observable, n_shots)

    params_minus = copy(params) .- c .* delta
    dispatch!(circuit, params_minus)
    fid_minus = sample(circuit, observable, n_shots)

    grad = (fid_plus - fid_minus) ./ (2 .* c .* delta)

    # Reset circuit to original parameters.
    dispatch!(circuit, params)
    return grad
end

function finite_diff_grad(f, x, delta = 0.0001)
    grad = zeros(length(x))
    @threads for i = 1:length(x)
        x_ = deepcopy(x)
        x_[i] += delta
        f1 = f(x_)
        x_[i] -= 2 * delta
        f2 = f(x_)
        grad[i] = (f1 - f2) / delta
    end
    return grad
end