"""Task module to define task context."""

import numpy as np
from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime, SingleShooting
import casadi as cs
from tasho import task_prototype_rockit as tp

# This may be replaced by some method get_ocp_variables, which calls self.states, self.controls, ...
from collections import namedtuple

_OCPvars = namedtuple("OCPvars", ["q", "q_dot", "q_ddot", "q0", "q_dot0"])


class Point2Point(tp.task_context):
    """Docstring for class Point2Point.

    This class defines a point-to-point motion problem
    by setting its constraints, objective and ocp variables.::

        from tasho import problem_template as pt
        problem = [t.Point2Point()

    """

    def __init__(self, time=None, horizon_steps=20, goal=None):
        # First call __init__ from TaskContext
        if time is None:
            super().__init__(horizon_steps=horizon_steps)
        else:
            super().__init__(time=time, horizon_steps=horizon_steps)

        if goal is not None:
            if (isinstance(goal, list) and int(len(goal)) == 3) or (
                isinstance(goal, np.ndarray) and goal.shape == (3, 1)
            ):
                print("Goal position")
            elif isinstance(goal, np.ndarray) and goal.shape == (4, 4):
                print("Goal transformation matrix")
        else:
            print(
                "ERROR: Please set a goal (position or transformation matrix) for the Point2Point problem"
            )

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
        finalposition_con = {
            "hard": True,
            "type": "Frame",
            "expression": fk_fun,
            "reference": goal,
        }
        finalvelocity_con = {"hard": True, "expression": q_dot, "reference": 0}
        final_constraints = {
            "final_constraints": [finalposition_con, finalvelocity_con]
        }

        # Add constraints to task
        self.add_task_constraint(final_constraints)

        self.add_regularization(expression=(fk_fun - goal), weight=1e-5, norm="L2")
        # self.add_regularization(
        #     expression=q, weight=1e-3, norm="L2", variable_type="state", reference=0
        # )
        self.add_regularization(
            expression=q_dot, weight=1e-3, norm="L2", variable_type="state", reference=0
        )
        self.add_regularization(
            expression=q_ddot,
            weight=1e-3,
            norm="L2",
            variable_type="control",
            reference=0,
        )

        # Set settings
        self.set_ocp_solver(SOLVER, SOLVER_SETTINGS)
        self.set_discretization_settings(DISC_SETTINGS)

        q0_val = robot.current_state[0 : robot.ndof]
        q_dot0_val = robot.current_state[robot.ndof :]

        self.set_value(q0, q0_val)
        self.set_value(q_dot0, q_dot0_val)
