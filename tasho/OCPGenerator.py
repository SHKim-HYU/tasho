# from ast import Expr
# from asyncio import current_task
from tasho import TaskModel, task_prototype_rockit as tp
import copy
import logging

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



    def append_task(self, task : TaskModel.Task, FreeTime, discretization_settings, exclude_continuity = None, include_continuity = None, generic_inter_stage_constraints = None):

        if FreeTime:
            stage = self.tc.create_stage(horizon_steps=discretization_settings["horizon_steps"])
        else:
            stage = self.tc.create_stage(time = discretization_settings["time_period"], horizon_steps=discretization_settings["horizon_steps"])
        self.stages.append(stage)
        self.stage_tasks.append(copy.deepcopy(task))
        self._generate_task_ocp(len(self.stages) - 1)

        # adding continuity and inter-stage constraints
        prev_stage = self.stages[-2]
        prev_stage_task = self.stage_tasks[-2]
        curr_stage_task = self.stage_tasks[-1]

        
        for var in task.variables:
            # Add continuity constraint to all states with same uid across consecutive tasks
            if task.variables[var].type == 'state' and task.variables[var] not in exclude_continuity and prev_stage_task:
                self.tc.ocp.subject_to(prev_stage.at_tf(prev_stage_task.variables[var].x) == stage.at_t0(curr_stage_task.variables[var].x))

        # Add continuity constraints on variable pairs in include continuity, but they must have different uids
        for var_pair in include_continuity:
            if task.variables[var_pair[1].uid] == prev_stage_task.variables[var_pair[0].uid]:
                logging.warning(f"Continuity constraint added by default for {var_pair[1].uid} and {var_pair[0].uid}, so not adding again")
            else:
                # adding the continuity constraint on the new variables
                self.tc.ocp.subject_to(prev_stage.at_tf(prev_stage_task.variables[var_pair[0]].x) == stage.at_t0(curr_stage_task[var_pair[1]].x))

        # Add generic inter-stage constraint
        # TODO: missing support for inter-stage constraint on expressions also, but do not see a big need.
        for con in generic_inter_stage_constraints:
            lambda_args = []
            for s in con[1]:
                lambda_args.append(prev_stage.at_tf(prev_stage_task.variables[s.uid]))
            for s in con[2]:
                lambda_args.append(curr_stage_task.at_t0(curr_stage_task.variables[s.uid]))

            expr = con[0](*lambda_args)
            if con[3] is not None:
                self.tc.ocp.subject_to(expr == con[3])
            elif con[4] is not None and con[5] is None:
                self.tc.ocp.subject_to(expr >= con[4])
            elif con[4] is None and con[5] is not None:
                self.tc.ocp.subject_to(expr <= con[5])
            elif con[4] is not None and con[5] is not None:
                self.tc.ocp.subject_to(con[4] <= (expr <= con[5]))
            
        return stage

    def _generate_task_ocp(self, stage_number):

        # Replacing task placeholder states with rockit placeholder states

        if stage_number > 0:
            stage_str = "_stage"+str(stage_number)
        else:
            stage_str = ""

        evaluated_expressions = set() #keep track of all expressions evaluated
        task = self.stage_tasks[stage_number]
        for var in task.variables.values():
            if var.type is not "magic_number":
                var._x = self.tc.create_expression(var.uid + stage_str, var.type, var.shape, stage_number)
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