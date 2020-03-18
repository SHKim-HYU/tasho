#helper functions using Tasho to set up the variables and parameters
#for many standard cases such as velocity-resolved, acceleration-resolved and
#Torque-resolved MPCs to simplify code

#import sys
from tasho import task_prototype_rockit as tp
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
import numpy as np

## acceleration_resolved(tc, robot, options):
# Function returns the expressions for acceleration-resolved control
# with appropriate position, velocity and acceleration constraints added
# to the task context
# @params tc The task context
# @params robot The object of the robot in question
# @params options Dictionary to pass further miscellaneous options 
def acceleration_resolved(tc, robot, options):

	print("ERROR: Not implemented")

def velocity_resolved(tc, robot, options):

	print("ERROR: Not implemented and probably not recommended")

def torque_resolved(tc, robot, options):

	print("ERROR: Not implemented")