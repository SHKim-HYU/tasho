"""Task module to define task context."""

import numpy as np
from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime, SingleShooting
import casadi as cs
from tasho import task_prototype_rockit as tp

# This may be replaced by some method get_ocp_variables, which calls self.states, self.controls, ...
from collections import namedtuple
_OCPvars = namedtuple("OCPvars", ['q', 'q_dot', 'q_ddot', 'q0', 'q_dot0'])


class Point2Point(tp.task_context):
    """Docstring for class Point2Point.

    This class defines a point-to-point motion problem
    by setting its constraints, objective and ocp variables.::

        from tasho import Problem
        problem = Point2Point()

    """
    def __init__(self, time = None, horizon = 10, goal = None):
        # First call __init__ from TaskContext
        if time is None:
        	super().__init__(horizon = horizon)
        else:
        	super().__init__(time = time, horizon = horizon)


        if goal is not None:
        	if ((isinstance(goal, list) and int(len(goal)) == 3) or
        		(isinstance(goal, np.ndarray) and goal.shape == (3, 1))):
        		print("Goal position")
        	elif (isinstance(goal, np.ndarray) and goal.shape == (4, 4)):
        		print("Goal transformation matrix")

        self.goal = goal

    def add_robot(self, robot):
        self.robots[robot.name] = robot
        # robot.transcribe(self)
        self.set_input_resolution(robot)

        # TODO: Rethink how much should be included in robot.transcribe method, since this should work for multiple robots
        # Maybe using task.transcribe instead of robot.transcribe is an option

        goal = self.goal

        # Based on Jeroen's example
        SOLVER = "ipopt"
        SOLVER_SETTINGS = {
            "ipopt": {
                "max_iter": 1000,
                "hessian_approximation": "limited-memory",
                "limited_memory_max_history": 5,
                "tol": 1e-3,
            }
        }
        DISC_SETTINGS = {
            "discretization method": "multiple shooting",
            "horizon size": self.horizon,
            "order": 1,
            "integration": "rk",
        }

        # Set the problem up
        q, q_dot, q_ddot, q0, q_dot0 = self.OCPvars

        # Set expression for End-effector's transformation matrix
        for _key in self.robots:
        	_robot = self.robots[_key]
        fk_fun = _robot.fk(q)[7]

        # Define pose constraints
        finalposition_con = {"hard": True, "type": "Frame", "expression": fk_fun, "reference": goal}
        finalvelocity_con = {"hard": True, "expression": q_dot, "reference": 0}
        final_constraints = {"final_constraints": [finalposition_con, finalvelocity_con]}

        # Define path constraints
        vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':1}
        acc_regularization = {'hard': False, 'expression':q_ddot, 'reference':0, 'gain':1}
        path_soft_constraints = {'path_constraints':[vel_regularization, acc_regularization]}

        # Add constraints to task
        self.add_task_constraint(final_constraints)
        self.add_task_constraint(path_soft_constraints)

        # Set settings
        self.set_ocp_solver(SOLVER, SOLVER_SETTINGS)
        self.set_discretization_settings(DISC_SETTINGS)
