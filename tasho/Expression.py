from tasho import Variable as Variable

class Expression(Variable):
    """
    A class defining the expression data-type, used for creating and re-using expressions.
    """

    def __init__(self, name, expression, mid = '', children = None):

        """
        Constructor for the expression. Overrides the constructor of the variable function.

        :param name: name of the expression. Should differentiate the expression from other expressions with the same meta id.
        :type name: String

        :param expression: The symbolic expression itself.
        :type expression: CasADi MX expression.

        :param mid: The meta-id of the expression. Optional, but highly recommended.
        :type mid: String

        :param children: A set of children expressions. Useful for keeping track of how an expression was built, while debugging.
        :type children: A set of either expressions of variables.
        """

        self._name = name
        self._x = expression
        self._mid = mid
        self._uid = mid + '_' + name

        self._children = children

        self._children_variables = set()
        for child in children:
            self._children_variables.union(child.children_variables)

    @property
    def children(self):
        return self._children