from tasho.Variable import Variable

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

        #TODO: shouldn't the reference to parents be in the task class instead?
        self._parents = parents
        self._x = expr_fun(*[p.x for p in parents]) #placeholder MX expression
        self._parent_uid = [x.uid for x in parents]

    def evaluate_expression(self, task):

        """
        Evaluates the expression function
        """
        args = []
        for arg in self._parent_uid:
            if arg in task.variables:
                args.append(task.variables[arg]._x)
            elif arg in task.expressions:
                args.append(task.expressions[arg]._x)
            else:
                raise Exception("Should not reach here")
        # args = [task.variables[arg.uid].x for arg in self._parents if arg.uid in task.variables]
        self._x = self._expr_fun(*args)
        
        return self._x

    @property
    def mid(self):
        return self._mid

    @property
    def name(self):
        return self._name

    @property 
    def uid(self):
        return self._uid

    @property
    def parents(self):
        return self._parents

    @property 
    def parent_uid(self):
        return self._parent_uid
