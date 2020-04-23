"""Problem module to define specific problems involving tasks."""

import sys
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np

class Problem:
    """Docstring for class Problem.

    This should be a description of the Problem class.
    It's common for programmers to give a code example inside of their
    docstring::

        from tasho import Problem
        problem = Problem()=

    Here is a link to :py:meth:`__init__`.
    """
    def __init__(self, name = "problem"):
        self.name = name
