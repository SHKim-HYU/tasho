from tasho import Variable, ConstraintExpression, Expression
import logging
import casadi as cs

class AbstractTask:
    """
    AbstractTask, the base class to build abstract tasks from. 
    """

    _name = 'AbstractClass'
    _symb = cs.MX.sym

    def __init__(self, tc, id):
        
        self._variables = {}
        self._expressions = {}
        self._constraint_expressions = {}
        self._constraints = {}
        self._sub_tasks = {}
        self._magic_numbers = {}
        self._id = id
        # self._tc = tc
        self._logger = logging.getLogger('AT_' + id)
        

    def create_variable(self, name, mid, type, shape):

        var = Variable(name, mid, type, shape)
        assert var.uid not in self._variables, name + " already used for a variable with the same meta-id."
        self._variables[var.uid] = var

        return var

    def substitute_variable(self):

        raise Exception("Not implemented")

    def remove_variable(self):

        raise Exception("Not implemented")

    def create_expression(self, name, mid, expression, *parents):
        
        expr = Expression(name, mid, expression, *parents)
        assert expr.uid not in self._expressions, name + " already used for an expression with the same meta-id."
        self._expressions[expr.uid] = expr

        return expr

    def remove_expression(self):

        raise Exception("Not implemented")

    def add_constraint_expression(self, name, mid, expression, constraint_hardness, **kwargs):

        con_expr = ConstraintExpression(name, mid, expression, constraint_hardness, **kwargs)
        assert con_expr.uid not in self._constraint_expressions, con_expr.uid + " already used for a constraint expression."
        self._constraint_expressions[con_expr.uid] = con_expr

        return con_expr

    def add_path_constraints(self, *args):

        """
        Impose all the constraints passed as parameters over the entire OCP horizon.
        """

        self._add_x_constraint("path", *args)

    def add_terminal_constraints(self, *args):

        """
        Impose the constraints passed as parameters only as the terminal constraints of the OCP.
        """

        self._add_x_constraints("terminal", *args)

    def add_initial_constraints(self, *args):

        """
        Impose the constraints passed as parameters as the initial constraints of the OCP.
        """

        self._add_x_constraints("initial", *args)

    def _add_x_constraint(self, x, *args):

        """
        Impose the constrainsts passed as the arguments as either path, terminal or initial constraints depending on x.
        """

        for arg in args:
            
            assert isinstance(arg, ConstraintExpression), "All arguments are expected to be constraint expressions."
            assert (x, arg.uid) not in self._constraints, "The imposed constraint is already present among constraints of the task."
            
            self._constraints[(x, arg.uid)] = (x, arg)

    def remove_initial_constraints(self, *args):

        raise Exception("Not implemented")

    def remove_path_constraints(self, *args):

        raise Exception("Not implemented")

    def remove_terminal_constraints(self, *args):

        raise Exception("Not implemented")

    def _remove_x_constraint(self, x, *args):

        raise Exception("Not implemented")
            

    def include_subtask(self, task2):

        """
        Mutator function that composes the constraints of task2 with the constraints of the object.
        Composition for the variables are based on IDs. If any variables or expressions have the same ID, they are assumed
        to refer to the same entity. 

        For tasks, at the moment all the tasks are directly added. There is no checking for duplicates. Only the IDs
        of each task should be unique.
        """
        
        assert task2.id not in self._sub_tasks, task2.id + " already one of the sub tasks."
        self._sub_tasks[task2.id]

        # Composing the variables.
        for var in task2.variables:

            if var.uid not in self.variables:
                self._variables[var.uid] = var
            else:
                self._logger.info("Ignoring the variable " + var.uid + " in task2 because a variable with an identical uid exists in task1.")
                var = self._variables[var.uid]

        # Composing the expressions
        for expr in task2.expressions:
            
            if expr.uid not in self._expressions:
                self._expressions[expr.uid] = expr
            else:
                self._logger.info("Ignoring the expression " + expr.uid + " in task2 because an expression with an identical uid exists in task1.")
                expr = self._expressions[expr.uid]


        # Composing the constraint expressions
        for con_expr in task2._constraint_expressions:

            if con_expr.uid not in self._constraint_expressions:
                self._constraint_expressions[con_expr.uid] = con_expr
                #TODO: Make a smarter composition rather than a blind intersection of all different constraints.
            else:
                self.logger.info("Ignoring the constraint expression in task2 because a constraint expression with the same uid exists in task1")
                con_expr = self._constraint_expressions[con_expr.uid]

        for cons in task2._constraints:

            if (cons[0], cons[1].uid) not in self._constraints:
                self._constraints[(cons[0], cons.uid)] = (cons[0], cons)
            else:
                raise Exception("Not implemented.")
        
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
