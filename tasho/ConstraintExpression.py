from tasho.Expression import Expression
from tasho.Variable import Variable

import casadi as cs

class ConstraintExpression:
    """
    A class defining the constraint expression object. This object forms the backbone of all the constraints added in the OCP.
    """

    def __init__(self, name, mid, expression, constraint_hardness, **kwargs):

        """
        A constructor for a constraint. Either reference, ub, lb or both lb and ub must be passed as an argument. If both reference and
        either/both ub and lb are passed, then reference is necessarily treated as a soft constraint while the ub and lb are
        treated as hard constraint. If soft constraint is imposed, the weight for penalizing constraint violation
        and optionally the type of the constraint violation penalty (by default quadratic penalty) must also
        be passed. If both ub and lb are passed, a box constraint is imposed. The constraint
        may be chosen to be hard or soft.

        :param name: Name of the constraint
        :type name: String

        :param expression: The expression on which the constraint is imposed. Preferably send an expression object.
        :type expression: Expression object or Casadi.MX type.

        :param constraint_hardness: 'hard' for hard constraint. 'soft' for soft constraint.
        :type constraint_hardness: String

        :param mid: An identifier from the code creating the constraint.
        :type mid: String

        :param reference: (optional, keyword argument) The reference that the expression must follow
        :type reference: CasADi DM, Python list or Numpy vector

        :param ub: (optional, keyword argument) The upper bound imposed on the expression.
        :type reference: CasADi DM, Python list or Numpy vector

        :param lb: (optional, keyword argument) The lower bound imposed on the expression.
        :type reference: CasADi DM, Python list or Numpy vector

        :param penalty: (optional, keyword argument, by default 'quad') Can be 'quad', 'L1' or 'L2'.
        :type penalty: String

        :param weight: (optional, keyword argument) Must be passed for soft constraints. It is the weight on the constraint violation penalty.
        :type weight: Double
        """
        assert isinstance(name, str)
        assert isinstance(expression, Expression) or isinstance(expression, Variable), "Unknown type passed as an Expression"
        assert constraint_hardness == 'hard' or constraint_hardness =='soft', "constraint_hardness must be either 'hard' or 'soft'"

        self._expr = expression.uid
        self._expression = expression
        con_dict = {}
        if 'reference' in kwargs:
            con_dict['equality'] = True
            con_dict['reference'] = kwargs['reference']
        elif 'lb' in kwargs and 'ub' in kwargs:
            con_dict['lub'] = True
            con_dict['upper_limits'] = kwargs['ub']
            con_dict['lower_limits'] = kwargs['lb']
        elif 'ub' in kwargs:
            con_dict['inequality'] = True
            con_dict['upper_limits'] = kwargs['ub']
        elif 'lb' in kwargs:
            raise Exception("Lower limits not implemented, please reformulate as upper limits.")
            self._symx = -self._symx
            con_dict['inequality'] = True
            con_dict['upper_limits'] = kwargs['upper_limits']
        else:
            raise Exception("Neither reference, lb or ub passed as constraint for the expression")

        
        if constraint_hardness == 'hard':
            con_dict['hard'] = True
        else:
            con_dict['hard'] = False

            assert 'weight' in kwargs, 'weight on the constraint violation penalty must be specified.'
            con_dict['gain'] = kwargs['weight']

            if 'penalty' not in kwargs:
                con_dict['penalty'] = 'quad'
            else:
                penalty = con_dict['penalty']
                assert penalty == 'quad' or penalty == 'L1' or penalty == 'L2', "Penalty must be 'quad', 'L1' or 'L2'"
                con_dict['norm'] = con_dict['penalty']

        self._constraint_dict = con_dict

        self._name = name
        self._mid = mid
        self._uid = mid + '_' + name

    def change_weight(self, new_weight):
        self._constraint_dict['gain'] = new_weight
    
    @property
    def mid(self):
        return self._mid

    @property
    def uid(self):
        return self._uid

    @property
    def expr(self):
        return self._expr

    @property
    def constraint_dict(self):
        return self._constraint_dict