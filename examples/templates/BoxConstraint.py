from tasho.ConstraintExpression import ConstraintExpression
from tasho.Expression import Expression
from tasho.Variable import Variable
import casadi as cs

def BoxConstraint(expression, lb, ub, name = None):

    """
    An abstract constraint, that returns a ConstraintExpression object. The returned constraint
    is a hard box constraint on the expression passed as the parameter. The mid of the constraint
    is by default "box_constraint_".

    :param expresssion: The expression on which the box constraint is imposed.
    :type expression: Expression or Variable object.

    :param lb: The lower limits of the box penalty.
    :type lb: CasADi DM, Python list or Numpy vector.

    :param ub: The upper limits of the box penalty.
    :type ub: CasADi DM, Python list or Numpy vector.

    :param name: (optional) By default, the uid of the expression.
    :type name: String
    """

    if name == None: name = expression.uid
    # if isinstance(ub, Variable) or isinstance(lb, Variable):
    #     box_expr = Expression(name, "limits_expr", lambda x, u, l : cs.vertcat(x- u, l - x), expression, ub, lb)
    #     con = ConstraintExpression(name, "limits", box_expr, "hard", ub = 0)
    # else:
    con = ConstraintExpression(name, "limits", expression, "hard", ub = ub, lb = lb)

    
    return con

