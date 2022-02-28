from ast import Expr
from tasho import task_prototype_rockit as tp
import copy

from tasho.Expression import Expression
from tasho.Variable import Variable

class OCPGenerator:

    def __init__(self, task, FreeTime, discretization_settings = None):
        """ 
        A constructor for the OCP generator object.   
        """

        self._solver_settings = {}
        if FreeTime:
            self.tc = tp.task_context(horizon_steps=discretization_settings["horizon_steps"])
        else:
            self.tc = tp.task_context(time = discretization_settings["time_period"], horizon_steps=discretization_settings["horizon_steps"])
        
        self.stages = [self.tc.stages[0]]
        self.stage_tasks = [copy.deepcopy(task)]
        self._generate_task_ocp(0)



    def append_task(self, task, FreeTime, discretization_settings):

        if FreeTime:
            stage = self.tc.create_stage(horizon_steps=discretization_settings["horizon_steps"])
        else:
            stage = self.tc.create_stage(time = discretization_settings["time_period"], horizon_steps=discretization_settings["horizon_steps"])
        self.stages.append(stage)
        self.stage_tasks.append(copy.deepcopy(task))
        return stage

    def _generate_task_ocp(self, stage_number):

        # Replacing task placeholder states with rockit placeholder states

        evaluated_expressions = set() #keep track of all expressions evaluated
        task = self.stage_tasks[stage_number]
        for var in task.variables.values():
            if var.type is not "magic_number":
                var._x = self.tc.create_expression(var.uid, var.type, var.shape, stage_number)
            evaluated_expressions |= set([var.uid])
        
        # Evaluating all expressions in a topological order via DFS
        for expr in task.expressions.values():

            if expr.uid not in evaluated_expressions:
                self._evaluate_expressions(expr, task, evaluated_expressions)
        
        # Add the constraints to the OCP 

        for cons in task.constraints:
            x = cons[0]
            con = copy.deepcopy(task.constraint_expressions[cons[1]])
            if con.expr in task.expressions:
                con.constraint_dict['expression'] = task.expressions[con.expr].x
            elif con.expr in task.variables:
                con.constraint_dict['expression'] = task.variables[con.expr].x

            # Converting other parameters of task constraints to their MX expressions
            dict_vals = ['upper_limits', 'lower_limits', 'reference', 'gain']
            for  d in dict_vals:
                if d in con.constraint_dict:
                    if isinstance(con.constraint_dict[d], Expression):
                        con.constraint_dict[d] = task.expresssions[con.constraint_dict[d].uid].x
                    elif isinstance(con.constraint_dict[d], Variable):
                        con.constraint_dict[d] = task.variables[con.constraint_dict[d].uid].x

            
            if x == "initial":
                self.tc.add_task_constraint({"initial_constraints":[con.constraint_dict]}, stage = stage_number)
            elif x == "path":
                self.tc.add_task_constraint({"path_constraints":[con.constraint_dict]}, stage = stage_number)
            elif x == "terminal":
                self.tc.add_task_constraint({"final_constraints":[con.constraint_dict]}, stage = stage_number)

        # setting the dynamics of the OCP
        for var in task.variables.values():
            if var.type == 'state':
                assert task._state_dynamics[var.uid][0] == 'der'
                var_der_uid = task._state_dynamics[var.uid][1]
                if var_der_uid in task.expressions:
                    self.tc.set_dynamics(var.x, task.expressions[var_der_uid].x, stage_number)
                elif var_der_uid in task.variables:
                    self.tc.set_dynamics(var.x, task.variables[var_der_uid].x, stage_number)
                else:
                    raise Exception("Should not reach here")


    def _evaluate_expressions(self, expr, task, exprs_evaluated):

        # verify that all parent expressions are evaluated
        for parent in expr._parent_uid:
            if parent not in exprs_evaluated:
                #if parent not evaluated, evaluate it
                self._evaluate_expressions(task._expressions[parent], task, exprs_evaluated)
        
        expr.evaluate_expression(task)
        


    

