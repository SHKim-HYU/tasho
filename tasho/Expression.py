from tasho import Variable as Variable

class Expression(Variable):
    """
    A class defining the expression data-type, used for creating and re-using expressions.
    """

    def __init__(self, name, mid, expr_fun, *parents):

        """
        Constructor for the expression. Overrides the constructor of the variable function.

        :param name: name of the expression. Should differentiate the expression from other expressions with the same meta id.
        :type name: String

        :param expression: The symbolic expression itself.
        :type expression: CasADi MX expression.

        :param mid: The meta-id of the expression. Optional, but highly recommended.
        :type mid: String

        :param parents: A set of parents expressions. Useful for keeping track of how an expression was built, while debugging. Necessary for removing variables and expressions from the task.
        :type parents: A set of either expressions of variables.
        """

        self._name = name
        self._expr_fun = expr_fun
        self._mid = mid
        self._uid = mid + '_' + name

        self._parents = parents
        self._x = self.evaluate_expression()

        #TODO: should this be in the task class instead?
        self._parent_variables = set()
        for parent in parents:

            if isinstance(parent, Expression):
                self._parent_variables.union(parent.parents_variables)
            elif isinstance(parent, Variable):
                self._parent_variables.union(parent)
            else:
                raise Exception("Must not reach here")

    def evaluate_expression(self):

        """
        Evaluates the expression function
        """

        args = [arg.x for arg in self._parents]
        x = self._expr_fun(*args)
        
        return x



    @property
    def parents(self):
        return self._parents