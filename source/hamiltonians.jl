using Yao
include("sketch.jl")

"Return Matrix of the TFIM Hamiltonian."
function tfim_ham(n, J, g)
    ham_zz = sum(kron(n, i => Z, i + 1 => Z) for i = 1:n-1)
    ham_x = sum(put(n, i => X) for i = 1:n)
    return applymatrix(J * ham_zz + g * ham_x)
end


"Return Matrix of the block TFIM Hamiltonian."
function block_tfim_ham(n_spins, n_blocks, J1, J2, g, return_matrix = true)
    n = n_blocks * n_spins
    ham = g * sum(put(n, i => X) for i = 1:n)
    for block = 1:n_blocks
        ham +=
            J1 * sum(
                kron(n, (block - 1) * n_spins + i => Z, (block - 1) * n_spins + i + 1 => Z)
                for i = 1:n_spins-1
            )
    end
    ham += J2 * sum(kron(n, b * n_spins => Z, b * n_spins + 1 => Z) for b = 1:n_blocks-1)
    if return_matrix
        return applymatrix(ham)
    else
        return ham
    end
end



"""Create a 2D hamiltonian from a sketch of the form
    g j g j g J g j g j g 
    j - - - j - j - - - j
    g j g j g J g j g j g

    where g indicates a spin with a transverse field, j and J are interactions between spins.
    and - indicates no interaction.
"""
function hamiltonian(sketch::Sketch)
    spins, edges = graph(sketch)
    return hamiltonian(spins, edges, sketch.values["g"])

end


"Create a hamiltonian from graph"
function hamiltonian(spins, edges, g)
    n = length(spins)
    ham = g * sum(put(n, spin => X) for spin in spins)
    for edge in edges
        ham += edge[2] * kron(n, edge[1][1] => Z, edge[1][2] => Z)
    end

    return applymatrix(ham)
end