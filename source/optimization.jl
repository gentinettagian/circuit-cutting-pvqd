"Handle termination criteria."
function terminate(history, step, opt_args)

    if !isnothing(opt_args.bin_size)
        if step < opt_args.bin_size
            return :continue
        end
        start = max(1, step - opt_args.bin_size + 1)
        loss = mean(history[start:step, 1])
        gnorm = mean(history[start:step, 2])

        if !isnothing(opt_args.rel_ftol) || !isnothing(opt_args.rel_gtol)
            if step - 1 < opt_args.bin_size
                return :continue
            end
            prev_loss = mean(history[start-1:step-1, 1])
            prev_gnorm = mean(history[start-1:step-1, 2])
        end

    else
        loss = history[step, 1]
        gnorm = history[step, 2]
        if !isnothing(opt_args.rel_ftol) || !isnothing(opt_args.rel_gtol)
            if step == 1
                return :continue
            else
                prev_loss = history[step-1, 1]
                prev_gnorm = history[step-1, 2]
            end
        end
    end

    if !isnothing(opt_args.abstol)
        if (loss < opt_args.abstol)
            return :abstol
        end
    end
    if !isnothing(opt_args.gradtol)
        if gnorm < opt_args.gradtol
            return :gradtol
        end
    end
    if !isnothing(opt_args.rel_ftol)
        if abs(loss - prev_loss) < opt_args.rel_ftol * abs(prev_loss)
            return :rel_ftol
        end
    end
    if !isnothing(opt_args.rel_gtol)
        if abs(gnorm - prev_gnorm) < opt_args.rel_gtol * abs(prev_gnorm)
            return :rel_gtol
        end
    end

    return :continue
end


"Construct an optimizer."
function construct(optimizer_args, params)
    # Define the optimizer
    if optimizer_args.method == :adam
        #optimizer = Optimisers.setup(Optimisers.ADAM(optimizer_args.lr, (0.9, 0.8)), params)
        optimizer = Optimisers.setup(
            Optimisers.Adam(optimizer_args.lr, optimizer_args.other),
            params,
        )
    elseif optimizer_args.method == :gd
        optimizer = Optimisers.setup(Optimisers.Descent(optimizer_args.lr), params)
    else
        error("Only Adam and Gradient Descent optimizer is implemented.")
    end
    return optimizer
end



