#helper functions using Tasho to set up the variables and parameters
#for many standard cases such as velocity-resolved, acceleration-resolved and
#Torque-resolved MPCs to simplify code

#import sys
from tasho import task_prototype_rockit as tp
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
import numpy as np


def acceleration_resolved(tc, robot, options):

	""" Function returns the expressions for acceleration-resolved control
	with appropriate position, velocity and acceleration constraints added
	to the task context
	:param name: tc The task context
	:param name: robot The object of the robot in question
	:param name: options Dictionary to pass further miscellaneous options 
	"""

	q = tc.create_expression('q', 'state', (robot.ndof, 1)) #joint positions over the trajectory
	q_dot = tc.create_expression('q_dot', 'state', (robot.ndof, 1)) #joint velocities
	q_ddot = tc.create_expression('q_ddot', 'control', (robot.ndof, 1))

	#expressions for initial joint position and joint velocity
	q0 = tc.create_expression('q0', 'parameter', (robot.ndof, 1))
	q_dot0 = tc.create_expression('q_dot0', 'parameter', (robot.ndof, 1))

	tc.set_dynamics(q, q_dot)
	tc.set_dynamics(q_dot, q_ddot)

	#add joint position, velocity and acceleration limits
	pos_limits = {'lub':True, 'hard': True, 'expression':q, 'upper_limits':robot.joint_ub, 'lower_limits':robot.joint_lb}
	vel_limits = {'lub':True, 'hard': True, 'expression':q_dot, 'upper_limits':robot.joint_vel_ub, 'lower_limits':robot.joint_vel_lb}
	acc_limits = {'lub':True, 'hard': True, 'expression':q_ddot, 'upper_limits':robot.joint_acc_ub, 'lower_limits':robot.joint_acc_lb}
	joint_constraints = {'path_constraints':[pos_limits, vel_limits, acc_limits]}
	tc.add_task_constraint(joint_constraints)

	#adding the initial constraints on joint position and velocity
	joint_init_con = {'expression':q, 'reference':q0}
	joint_vel_init_con = {'expression':q_dot, 'reference':q_dot0}
	init_constraints = {'initial_constraints':[joint_init_con, joint_vel_init_con]}
	tc.add_task_constraint(init_constraints)


	return q, q_dot, q_ddot, q0, q_dot0

def velocity_resolved(tc, robot, options):

	print("ERROR: Not implemented and probably not recommended")

def torque_resolved(tc, robot, options):

	print("ERROR: Not implemented")