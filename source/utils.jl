using Yao
using Plots

"Fidelity between two states."
function fidelity(state1, state2)
    return abs2(dot(state1', state2))
end

"Flatten a circuit"
function flatten(circuit::ChainBlock; keep_rzz_gates = true)
    flattened = chain(circuit.n)
    function add_or_continue(gate)
        if gate isa ChainBlock
            if length(gate) == 0
                return
            end
            if keep_rzz_gates && gate[1] isa ControlBlock
                push!(flattened, gate)
                return
            end
            for g in gate
                add_or_continue(g)
            end
        else
            push!(flattened, gate)
        end
    end
    for gate in circuit
        add_or_continue(gate)
    end
    return flattened
end


zero_proj(n) = kron([0.5 * (I2 + Z) for i = 1:n]...)
local_proj(n) = 1 / n * sum(kron([i == j ? 0.5 * (I2 + Z) : I2 for i = 1:n]...) for j = 1:n)



"Implementing the `Rzz` rotation gate."
function Rzz(n, i, j, theta)
    circ = chain(n, [cnot(i, j), put(j => Rz(theta)), cnot(i, j)])
    return circ
end

"Implements a full rotation"
function rxy(theta, phi)
    return chain(1, [Rx(theta), Ry(phi)])
end


"Overhead due to circuit cutting."
function gamma(theta)
    prod(1 .+ 2 * sqrt.(sin.(theta) .^ 2)) .^ 4
end


"Get result from path"
function get_result(path::String)
    return load_object(path * "/result")
end

"Get options from path"
function get_options(path::String)
    return load_object(path * "/setup_args.txt")
end



font_name = font(24, "Computer Modern")
small_font = font(12, "Computer Modern")
normal_font = font(12, "Computer Modern")
large_font = font(13, "Computer Modern")
footnotesize_font = font(11, "Computer Modern")


"Plot template"
function gplot(xlabel, ylabel, options = nothing; ratio = 1 / 1.62, width = 426.79135)
    if isnothing(options)
        options = Dict("yaxis" => :left, "legend" => :bottomright)
    end
    if options["yaxis"] == :left
        ymirror = false
    else
        ymirror = true
    end


    p = Plots.plot(
        xlabel = xlabel,
        ylabel = ylabel,
        legend = options["legend"],
        size = (width, width * ratio),
        dpi = 300,
        legendfont = small_font,
        tickfont = footnotesize_font,
        xlabelfont = small_font,
        ylabelfont = small_font,
        framestyle = :box,
        ymirror = ymirror,
        guidefont = small_font,
        titlefont = small_font,
        #margin = 2.5mm,
        #widen = false,
    )
    return p
end


"Get statevector from circuit."
function statevector(circ::ChainBlock)
    return Yao.state(zero_state(circ.n) |> circ)
end