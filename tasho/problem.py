"""Problem module to define specific problems involving tasks."""

from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp, FreeTime
import numpy as np

class Problem:
    """Docstring for class Problem.

    This should be a description of the Problem class.
    It's common for programmers to give a code example inside of their
    docstring::

        from tasho import Problem
        problem = Problem()

    Here is a link to :py:meth:`__init__`.
    """
    def __init__(self, name = "problem", T_goal = None, p_goal = None, R_goal = None):
        self.name = name
        self.task_context = None
        self.robots = []
        
        self.T_init = 10.0
        # self.ocp = Ocp(T=FreeTime(self.T_init))

    def add_robot(self, robot):
        self.robots.append(robot)
        robot.transcribe(self.task_context)
        self.sim_system_dyn = robot.sim_system_dyn(self.task_context)



class Point2Point(Problem):
    """Docstring for class Point2Point.

    This class defines a point-to-point motion problem
    by setting its constraints, objective and ocp variables.::

        from tasho import Problem
        problem = Point2Point()

    """
