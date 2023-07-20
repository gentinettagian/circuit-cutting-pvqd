using Yao, YaoExtensions, Yao.AD

zero_proj(n) = kron([0.5 * (I2 + Z) for i = 1:n]...)

"Calculate the fidelity between two circuits."
function fidelity(circ_1::ChainBlock, circ_2::ChainBlock)
    compute_uncompute = chain(circ_1, circ_2')
    observable = zero_proj(circ_1.n)
    fidelity = real(expect(observable, zero_state(circ_1.n) |> compute_uncompute))
    return fidelity
end

"Calculate the gradient of the fidelity."
function grad_fidelity(circ_1, circ_2)
    compute_uncompute = chain(circ_1, circ_2')
    observable = zero_proj(circ_2.n)
    _, grad = expect'(observable, zero_state(circ_2.n) => compute_uncompute)
    return grad
end


