from numpy import isin
from tasho.Variable import Variable
from tasho.ConstraintExpression import ConstraintExpression
from tasho.Expression import Expression
import logging
import casadi as cs
import networkx as nx
import pydot

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

        elif isinstance(entity[0], str) and isinstance(entity[1], str):
            self.graph.add_node(entity)
            self.graph.add_edge(entity[1], entity)

    def _remove_from_graph(self, entity):
        """
        Recursively removes the expression and all its descendents from the graph
        """
        children =  list(self.graph.successors(entity))
        for child in children:
            if self.graph.has_node(child): self._remove_from_graph(child)

        self.graph.remove_node(entity)

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
        
    def substitute_expression(self, old_var, new_var):

        """ 
        Substitutes any expression/variable in the task with another expression/variable. 

        :param old_var: The expression being replaced.
        :type old_var: tasho.Expression or tasho.Variable

        :param new_var: The replacing expression
        :type new_var: tasho.Expression or tasho.Variable
        """

        assert old_var.shape == new_var.shape, "Attempting to substitute variable with a variable of different shape"

        # remove the derivative of the old variable
        if isinstance(old_var, Variable): self._state_dynamics.pop(old_var.uid)

        # assign the new variable as the derivative of other expressions that had the old variable as the derivative
        for x in self._state_dynamics.values():
            if x[1] == old_var.uid: x[1] = new_var.uid

        # Assign the new var to all the children of the old variable
        for c in self.graph.successors(old_var.uid): 
            i = self._expressions[c]._parent_uid.index(old_var.uid)
            self._expressions[c]._parent_uid[i] = new_var.uid
            self.graph.add_edge(new_var.uid, self._expressions[c].uid) # adding a new connection in the graph

        self.add_expr_recursively(new_var) # adding all the ancestors of new_var recursively
        
        if isinstance(old_var, Variable): self._variables.pop(old_var.uid) #deleting the old variable
        self._expressions.pop(old_var.uid)

        #remove the old node from the graph
        self.graph.remove_node(old_var.uid)

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
            for p in expr._parents: self.add_expr_recursively(p)
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

    def remove_constraint_expression(self, expr):

        assert expr.uid in self.constraint_expressions

        con_types = ["initial", "path", "terminal"]
        for con in con_types:
            if (con, expr.uid) in self._constraints:
                self._remove_x_constraint(con, expr)

        self._constraint_expressions.pop(expr.uid)
        
        self._remove_from_graph(expr.uid)

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
            self.add_constraint_expression(arg)
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
        list(map(self.add_variable, task2._variables.values()))

        # Composing the expressions
        list(map(self.add_expression, task2._expressions.values()))
        
        # Composing the constraint expressions
        #TODO: Make a smarter composition rather than a blind intersection of all different constraints.
        list(map(self.add_constraint_expression, task2._constraint_expressions.values()))

        # Composing constraints
        for cons in task2._constraints:

            if cons not in self._constraints:
                self._constraints[cons] = cons
                self._add_to_graph(cons)
            else:
                raise Exception("Not implemented.")

    def write_task_graph(self, name):
        """
        Write the task graph in to an SVG file saved at the file location.

        :param name: The argument should pass the SVG file name and the location in the following format "file_location/filename.svg" 
        :type name: string
        """
        
        graph = nx.drawing.nx_pydot.to_pydot(self.graph)

        # set all the expression nodes to ellipses with green color
        for expr in self._expressions:
            node = graph.get_node(expr)[0]
            node.set_shape('ellipse')
            node.set_color('green')

        # set all the variable nodes to circles and assign color based on the type
        for var in self._variables.values():
            node = graph.get_node(var.uid)[0]
            node.set_shape("doubleoctagon")
            type_to_color = {"state":"green", "control":"cyan", "parameter":"blue", "variable":"yellow", "magic_number":"red"}
            node.set_color(type_to_color[var.type])

        # set all the constraint expressions to hexagons
        for var in self._constraint_expressions:
            node = graph.get_node(var)[0]
            node.set_shape("hexagon")
            node.set_color("magenta")

        # set all the constraints to boxes
        for var in self._constraints:
            node = graph.get_node(f'"(\'{var[0]}\', \'{var[1]}\')"')[0]
            node.set_shape("box")
            node.set_color("purple")

        for var in self._state_dynamics:
            graph.add_edge(pydot.Edge(self._state_dynamics[var][1], var, edge_color = 'violet', style = 'dashed'))

        assert name[-4:] == '.svg', "The file format should be .svg"
        graph.write_svg(name)
        
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
