from numpy import isin
from tasho.Variable import Variable
from tasho.ConstraintExpression import ConstraintExpression
from tasho.Expression import Expression
import logging
import casadi as cs
import networkx as nx

class Task:
    """
    Task, the base class to build abstract tasks from. 
    """

    _name = 'AbstractClass'
    _symb = cs.MX.sym

    def __init__(self, name, mid):
        
        self._name = name
        self._mid = mid
        self._uid = mid + '_' + name

        self._variables = {}
        self._state_dynamics = {} #for state variables, stores an expression for either continuous-time or a discrete-time dynamics
        self._expressions = {}
        self._constraint_expressions = {}
        self._constraints = {}
        self._sub_tasks = {}
        self._magic_numbers = {}
        self._monitors = {}
        self._logger = logging.getLogger('AT_' + self._uid)
        self.graph = nx.DiGraph()
        
    def _add_to_graph(self, entity):
        """
        Adds a variable, expression, constraint expression or a constraint to the graph.
        """
        if isinstance(entity, Variable):
            self.graph.add_node(entity.uid)
            for x in entity._parent_uid: self.graph.add_edge(x, entity.uid)

        elif isinstance(entity, ConstraintExpression):
            self.graph.add_node(entity.uid)
            self.graph.add_edge(entity.expr, entity.uid)

        elif isinstance(entity, (str, str)):
            self.graph.add_node(entity)
            self.graph.add_edge(entity[1], entity)

    def create_variable(self, name, mid, type, shape):

        var = Variable(name, mid, type, shape)
        assert var.uid not in self._variables, name + " already used for a variable with the same meta-id."
        self._variables[var.uid] = var

        assert var.uid not in self._expressions, name + " already used for an expression"
        self._expressions[var.uid] = var

        self._add_to_graph(var)

        return var

    def add_variable(self, var):

        if var.uid not in self._variables:
            self._variables[var.uid] = var
            self._add_to_graph(var)
            return
        self._logger.info("Not adding variable " + var.uid + " because a variable with identical uid already exists.")
        
    def substitute_variable(self, old_var, new_var):
        """ TODO: Recheck """
        assert old_var.type == new_var.type, "Attempting to substitute variable with a variable of wrong type"
        assert old_var.shape == new_var.shape, "Attempting to substitute variable with a variable of different shape"
        self._variables[old_var.uid] = new_var
        self._expressions[old_var.uid] = new_var
        new_var._uid = old_var.uid

    def remove_variable(self):

        raise Exception("Not implemented")

    def set_der(self, var, dyn):

        """
        Sets the derivative of a given state variable to a given expression.

        :param var: The state variable whose derivative is being set.
        :type var: tasho.Variable of type "state"

        :param dyn: The expression for the derivative of var.
        :type var: tasho.Variable or tasho.Expression
        """

        assert var.type == 'state', "Attempting to set derivative to a non-state variable"
        assert isinstance(var, Variable)
        assert var.uid in self._variables, "Attempting to assign derivative to a state that does not exist in the task."
        assert isinstance(dyn, Variable), "The derivative is not an expression"
        assert dyn.uid in self._expressions or dyn.uid in self._variables, "The derivative expression does not exist in the task."

        self._state_dynamics[var.uid] = ['der', dyn.uid]


    def set_next(self, var, dyn):

        raise Exception("Not implemented")


    def create_expression(self, name, mid, expression, *parents):
        
        expr = Expression(name, mid, expression, *parents)
        assert expr.uid not in self._expressions, name + " already used for an expression with the same meta-id."
        self._expressions[expr.uid] = expr
        self._add_to_graph(expr)
        return expr

    def add_expression(self, expr):
        if expr.uid not in self._expressions:
            self._expressions[expr.uid] = expr
            self._add_to_graph(expr)
        else:
            self._logger.info("Not adding expression " + expr.uid + " because an expression with identical uid already exists.")

    def add_expr_recursively(self, expr):

        if expr.uid in self._expressions or expr.uid in self._variables: return
        
        if isinstance(expr, Expression): 
            map(self.add_expr_recursively, expr._parents)
            self.add_expression(expr)
        elif isinstance(expr, Variable): self.add_variable(expr)
        else: raise Exception("Must not reach here!")

    def remove_expression(self):

        raise Exception("Not implemented")

    def create_constraint_expression(self, name, mid, expression, constraint_hardness, **kwargs):

        con_expr = ConstraintExpression(name, mid, expression, constraint_hardness, **kwargs)
        assert con_expr.uid not in self._constraint_expressions, con_expr.uid + " already used for a constraint expression."
        self._constraint_expressions[con_expr.uid] = con_expr
        self._add_to_graph(con_expr)
        return con_expr

    def add_constraint_expression(self, expr):

        if expr.uid not in self._constraint_expressions:
            self._constraint_expressions[expr.uid] = expr
            self._add_to_graph(expr)
        else:
            self._logger.info("Not adding constraint expression " + expr.uid + " because a constraint expression with an identical uid exists.")

    def add_path_constraints(self, *args):

        """
        Impose all the constraints passed as parameters over the entire OCP horizon.
        """

        self._add_x_constraint("path", *args)

    def add_terminal_constraints(self, *args):

        """
        Impose the constraints passed as parameters only as the terminal constraints of the OCP.
        """

        self._add_x_constraint("terminal", *args)

    def add_initial_constraints(self, *args):

        """
        Impose the constraints passed as parameters as the initial constraints of the OCP.
        """

        self._add_x_constraint("initial", *args)

    def _add_x_constraint(self, x, *args):

        """
        Impose the constrainsts passed as the arguments as either path, terminal or initial constraints depending on x.
        """

        for arg in args:
            
            assert isinstance(arg, ConstraintExpression), "All arguments are expected to be constraint expressions."
            assert (x, arg.uid) not in self._constraints, "The imposed constraint is already present among constraints of the task."
            
            if arg.expr not in self._expressions:
                self.add_expr_recursively(arg._expression)
            self._constraints[(x, arg.uid)] = (x, arg.uid)
            self._add_to_graph((x, arg.uid))

    def remove_initial_constraints(self, *args):

        """
        Remove the initial constraints passed as arguments.
        """

        self._remove_x_constraint("initial", *args)

    def remove_path_constraints(self, *args):

        """
        Remove the path constraints passed as arguments.
        """

        self._remove_x_constraint("path", *args)

    def remove_terminal_constraints(self, *args):

        """
        Remove terminal constraints passed as arguments.
        """

        self._remove_x_constraint("path", *args)

    def _remove_x_constraint(self, x, *args):

        for arg in args:
            assert isinstance(arg, ConstraintExpression), "All arguments must be constraint expressions."
            assert (x, arg.uid) in self._constraints, "Attempting to remove a constraint not present."

            self._constraints.pop((x, arg.uid))          

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
        map(self.add_variable, task2._variables.values())

        # Composing the expressions
        map(self.add_expression, task2._expressions.values())
        
        # Composing the constraint expressions
        #TODO: Make a smarter composition rather than a blind intersection of all different constraints.
        map(self.add_constraint_expression, task2._constraint_expressions.values())

        # Composing constraints
        for cons in task2._constraints:

            if cons not in self._constraints:
                self._constraints[cons] = cons
                self._add_to_graph(cons)
            else:
                raise Exception("Not implemented.")
        
    @property
    def id(self):
        return self._id

    @property
    def variables(self):
        return self._variables
        
    @property
    def expressions(self):
        return self._expressions

    @property
    def constraints(self):
        return self._constraints
    
    @property
    def constraint_expressions(self):
        return self._constraint_expressions


def compose(self, name, mid, *args):

    """ 
    Creates a new task by composing all the tasks that are passed as arguments to the function.
    and returns this new task.
    """

    task = Task(name, mid)

    for arg in args:
        task.compose_within(arg)

    return task 
