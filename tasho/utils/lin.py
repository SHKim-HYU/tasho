from casadi import Function, MX, symvar, veccat

def lin(expr):
    # Get list of symbols s present in expression expr(s).
    s = symvar(expr)
    # Convert list into a column-vector of symbols x.
    # If the size of the symbols present in the expression expr is not 1x1, vertcat will not be able to create a column-vector of symbols.
    # That's why we use veccat instead of vertcat
    x = veccat(*s)
    # Define variable x0 with same sparsity as x.
    x0 = MX.sym('x0',x.sparsity())
    # Define function that outputs the expression expr wrt a vector of symbols x.
    F = Function('F',[x],[expr])
    # Get a function that calculates 1st forward derivatives. Forward mode AD for F
    # The first input corresponds to nondifferentiated inputs,
    # The next input corresponds to nondifferentiated outputs, 
    # and last input corresponds to forward seeds, stacked horizontally. 
    # It outputs the forward sensitivities, stacked horizontally.
    Ff = F.forward(1)
    # Get symbolic expression expr wrt x0
    n = F(x0)
    # Define symbolic expression of the original expression evaluated in x0 plus the forward sensitivity to build a first-order Taylor series approximation around x = x0
    expr_lin = n + Ff(x0,n,x-x0)
    # Define function for expr_lin indicating that x0 should be kept constant.
    # is_diff_in: Indicates for each input if it should be differentiable.
    # never_inline: Forbids inlining.
    FS = Function('FS', [x,x0], [expr_lin], {'is_diff_in':[True, False],'never_inline':True})
    # Return FS(x,x), which would be equivalent to evaluating F(x), but its second derivative is zero. 
    return FS(x,x)