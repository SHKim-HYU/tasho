from cmath import exp
from tasho.ConstraintExpression import ConstraintExpression

def Regularization(expression, weight, reference = 0):

    con = ConstraintExpression(expression.uid, "reg", expression, "soft", reference = reference, weight = weight)
    return con