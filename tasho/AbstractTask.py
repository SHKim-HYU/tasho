from tasho import Variable, ConstraintExpression, Expression

class AbstractTask:
    """
    AbstractTask, the base class to build abstract tasks from. 
    """

    _name = 'AbstractClass'

    def __init__(self, tc, id):
        
        self._variables = {}
        self._expressions = {}
        self._constraint_expressions = {}
        self._constraints = {}
        self._sub_tasks = {}
        self._id = id + '_' + self._name
        self._tc = tc
        

    def add_variable(self, name, type, shape):

        var = Variable(name, type, shape, self._tc, self._id)
        assert name not in self._variables, name + " already used for a variable in the task."
        self._variables[name] = var

        return var

    def remove_variable(self):

        raise Exception("Not implemented")

    def add_expression(self, name, expression, children = None):
        
        expr = Expression(name, expression, id = self._id, children = children)
        assert name not in self._expressions, name + " already used for an expression in the task."
        self._expressions[name] = expr

        return expr

    def remove_expression(self):

        raise Exception("Not implemented")

    def add_constraint_expression(self, name, expression, constraint_hardness, **kwargs):

        con_expr = ConstraintExpression(name, expression, constraint_hardness, **kwargs)
        assert name not in self._constraint_expressions, name + " already used for a constraint expression."
        self._constraint_expressions[name] = con_expr

        return con_expr

    def add_path_constraint(self):

        raise Exception("Not implemented")

    def add_terminal_constraint(self):

        raise Exception("Not implemented")

    def add_initial_constraint(self):

        raise Exception("Not implemented")

    def compose_within(self, task2):

        """
        Mutator function that composes the constraints of task2 with the constraints of the object.
        """
        
        assert task2.id not in self._sub_tasks, task2.id + " already one of the sub tasks."
        self._sub_tasks[task2.id]

        # Composing the variables.
        for var in task2.variables:

            assert var.id not in self.variables
            self._variables[var.id] = var

        # Composing the expressions
        for expr in task2.variables:
            
            assert expr.id not in self._expressions
            self._expressions[expr.id] = expr

        # Composing the constraint expressions
        for con_expr in task2._constraint_expressions:

            assert con_expr.id not in self._constraint_expressions
            self._constraint_expressions[con_expr.id] = con_expr

        for cons in task2._constraints:

            assert cons.id not in self._constraints
            self._constraints[cons.id] = cons

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str), "Non-string type passed as a name"
        self._name = name
        
    @property
    def id(self):
        return self._id

    @property
    def variables(self):
        return self._variables

    @property
    def constraints(self):
        return self._constraints
    
    @property
    def constraint_expressions(self):
        return self._constraint_expressions


def compose(self, id, tc, *args):

    """ 
    Creates a new task by composing all the tasks that are passed as arguments to the function.
    and returns this new task.
    """

    task = AbstractTask(tc, id)

    for arg in args:
        task.compose_within(arg)

    return task 
