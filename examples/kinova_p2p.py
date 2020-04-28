## OCP for point-to-point motion of a kinova robot arm

import sys
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

	print("Task specification and visualization of P2P OCP")

	horizon_size = 10
	t_mpc = 0.5
	max_joint_acc = 30*3.14159/180

	robot = rob.Robot('kinova')

	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)

	tc = tp.task_context(horizon_size*t_mpc)

	q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	#computing the expression for the final frame
	fk_vals = robot.fk(q)[7]

	final_pos = {'hard':True, 'expression':fk_vals[0:3, 3], 'reference':[0.5, 0.0, 0.5]}
	final_vel = {'hard':True, 'expression':q_dot, 'reference':0}
	final_constraints = {'final_constraints':[final_pos, final_vel]}
	tc.add_task_constraint(final_constraints)

	#adding penality terms on joint velocity and position
	vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':1}
	acc_regularization = {'hard': False, 'expression':q_ddot, 'reference':0, 'gain':1}

	task_objective = {'path_constraints':[vel_regularization, acc_regularization]}
	tc.add_task_constraint(task_objective)

	tc.set_ocp_solver('ipopt')
	tc.ocp.set_value(q0, [0]*7)
	tc.ocp.set_value(q_dot0, [0]*7)
	disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	tc.set_discretization_settings(disc_settings)
	sol = tc.solve_ocp()

	ts, q_sol = sol.sample(q, grid="control")
	print(q_sol)
	print(robot.fk(q_sol[-1,:])[7])
