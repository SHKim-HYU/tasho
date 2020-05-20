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
import time

if __name__ == '__main__':


	robot = rob.Robot('iiwa7')

	from tasho import world_simulator, MPC

	obj = world_simulator.world_simulator()

	
	position = [0.0, 0.0, 0.0]
	orientation = [0.0, 0.0, 0.0, 1.0]
	kukaID = obj.add_robot(position, orientation, 'iiwa7')
	max_joint_acc = 30*3.14159/180

	robot = rob.Robot('iiwa7')	
	
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)

	#obj.run_simulation(1000)

	horizon_size = 10
	t_mpc = 0.5


	tc = tp.task_context(horizon_size*t_mpc)

	q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	fk_vals = robot.fk(q)[7]

	T_goal = np.array([[0.0, 0., -1., 0.5], [0., 1., 0., -0.25], [1.0, 0., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]])

	final_pos = {'hard':True, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal}
	final_vel = {'hard':True, 'expression':q_dot, 'reference':0}
	final_constraints = {'final_constraints':[final_pos, final_vel]}
	tc.add_task_constraint(final_constraints)

	vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':1}
	acc_regularization = {'hard': False, 'expression':q_ddot, 'reference':0, 'gain':1}
	task_objective = {'path_constraints':[vel_regularization, acc_regularization]}
	tc.add_task_constraint(task_objective)

	tc.ocp.set_value(q0, [0]*7)
	tc.ocp.set_value(q_dot0, [0]*7)

	tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
	disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	tc.set_discretization_settings(disc_settings)
	sol = tc.solve_ocp()

	joint_indices = [0, 1, 2, 3, 4, 5, 6]

	no_samples = int(t_mpc/obj.physics_ts)
	mpc_params = {'world':obj}
	q0_params_info = {'type':'joint_position', 'joint_indices':joint_indices, 'robotID':kukaID}
	q_dot0_params_info = {'type':'joint_velocity', 'joint_indices':joint_indices, 'robotID':kukaID}
	mpc_params['params'] = {'q0':q0_params_info, 'q_dot0':q_dot0_params_info}

	mpc_params['disc_settings'] = disc_settings
	# 	# mpc_params['solver_name'] = 'ipopt'
	# 	# mpc_params['solver_params'] = {'lbfgs':True}
	# mpc_params['solver_name'] = 'sqpmethod'
	# 	# mpc_params['solver_params'] = {'qpoases':True}
	# 	# mpc_params['solver_params'] = {'qrqp':True}
	# 	# mpc_params['solver_params'] = {'osqp':True}
	# mpc_params['solver_params'] = {'ipopt':True}
	mpc_params['t_mpc'] = t_mpc
	mpc_params['control_type'] = 'joint_velocity' #
	mpc_params['control_info'] = { 'robotID':kukaID, 'discretization':'constant_acceleration', 'joint_indices':joint_indices, 'no_samples':no_samples}


	tc.add_monitor({"name":"termination_criteria", "expression":tc.ocp.objective, "reference":0.01, "lower":True, "initial":True})
	sim_type = "bullet_notrealtime"

	mpc_obj = MPC.MPC(tc, sim_type, mpc_params)

	#run the ocp with IPOPT to get a good initial guess for the MPC
	mpc_obj.configMPC_fromcurrent()

	# 	#run the MPC
	mpc_obj.runMPC()

	# 	# #begin the visualization of applying OCP solution in open loop
	# ts, q_dot_sol = sol.sample(q_dot, grid="control")
	# obj.resetJointState(kukaID, joint_indices, [0]*7)
	# obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[0])
	# obj.run_simulation(480)

	# no_samples = int(t_mpc/obj.physics_ts)
	# for i in range(horizon_size):
	# 	q_vel_current = 0.5*(q_dot_sol[i] + q_dot_sol[i+1])
	# 	obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_vel_current)
	# 	obj.run_simulation(no_samples)

	# obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[-1])

	# obj.run_simulation(no_samples)


	# print("Random bin picking with KUKA Iiwa")

	# visualizationBullet = True
	# horizon_size = 10
	# t_mpc = 0.5
	# 
	# print(robot.joint_name)
	# print(robot.joint_ub)
	# tc = tp.task_context(horizon_size*t_mpc)

	# q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	# #computing the expression for the final frame
	# print(robot.fk)
	# fk_vals = robot.fk(q)[7]

	# T_goal = np.array([[0.0, 0., -1., 0.7], [0., 1., 0., -0.25], [1.0, 0., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
	# final_pos = {'hard':True, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal}
	# final_vel = {'hard':True, 'expression':q_dot, 'reference':0}
	# final_constraints = {'final_constraints':[final_pos, final_vel]}
	# tc.add_task_constraint(final_constraints)

	# #adding penality terms on joint velocity and position
	# vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':1}
	# acc_regularization = {'hard': False, 'expression':q_ddot, 'reference':0, 'gain':1}

	# task_objective = {'path_constraints':[vel_regularization, acc_regularization]}
	# tc.add_task_constraint(task_objective)

	# tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
	# q0_val = [0]*7
	# tc.ocp.set_value(q0, q0_val)
	# tc.ocp.set_value(q_dot0, [0]*7)
	# disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	# tc.set_discretization_settings(disc_settings)
	# sol = tc.solve_ocp()

	# ts, q_sol = sol.sample(q, grid="control")
	# print(q_sol)
	# print(robot.fk(q_sol[-1,:])[6])


	# if visualizationBullet:

	# 	from tasho import world_simulator, MPC
	# 	import pybullet as p

	# 	obj = world_simulator.world_simulator()

	# 	position = [0.0, 0.0, 0.0]
	# 	orientation = [0.0, 0.0, 0.0, 1.0]

	# 	kukaID = obj.add_robot(position, orientation, 'iiwa7')
	# 	#Add a cylinder to the world
	# 	#cylID = obj.add_cylinder(0.15, 0.5, 0.5, {'position':[0.5, 0.0, 0.25], 'orientation':[0.0, 0.0, 0.0, 1.0]})
	# 	cylID = p.loadURDF("cube_small.urdf", [0.5, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 3.0)
	# 	#print(obj.getJointInfoArray(kukaID))
	# 	no_samples = int(t_mpc/obj.physics_ts)

	# 	if no_samples != t_mpc/obj.physics_ts:
	# 		print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

	# 	#correspondence between joint numbers in bullet and OCP determined after reading joint info of YUMI
	# 	#from the world simulator
	# 	joint_indices = [0, 1, 2, 3, 4, 5, 6]

	# 	# #begin the visualization of applying OCP solution in open loop
	# 	# ts, q_dot_sol = sol.sample(q_dot, grid="control")
	# 	# obj.resetJointState(kukaID, joint_indices, q0_val)
	# 	# obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[0])
	# 	# obj.run_simulation(480)


	# 	# for i in range(horizon_size):
	# 	# 	q_vel_current = 0.5*(q_dot_sol[i] + q_dot_sol[i+1])
	# 	# 	obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_vel_current)
	# 	# 	obj.run_simulation(no_samples)

	# 	# obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[-1])

	# 	mpc_params = {'world':obj}
	# 	q0_params_info = {'type':'joint_position', 'joint_indices':joint_indices, 'robotID':kukaID}
	# 	q_dot0_params_info = {'type':'joint_velocity', 'joint_indices':joint_indices, 'robotID':kukaID}
	# 	s0_params_info = {'type':'progress_variable', 'state':True}
	# 	s_dot0_params_info = {'type':'progress_variable', 'state':True}
	# 	mpc_params['params'] = {'q0':q0_params_info, 'q_dot0':q_dot0_params_info}

	# 	mpc_params['disc_settings'] = disc_settings
	# 	# mpc_params['solver_name'] = 'ipopt'
	# 	# mpc_params['solver_params'] = {'lbfgs':True}
	# 	mpc_params['solver_name'] = 'sqpmethod'
	# 	# mpc_params['solver_params'] = {'qpoases':True}
	# 	# mpc_params['solver_params'] = {'qrqp':True}
	# 	# mpc_params['solver_params'] = {'osqp':True}
	# 	mpc_params['solver_params'] = {'ipopt':True}
	# 	mpc_params['t_mpc'] = t_mpc
	# 	mpc_params['control_type'] = 'joint_velocity' #
	# 	mpc_params['control_info'] = { 'robotID':kukaID, 'discretization':'constant_acceleration', 'joint_indices':joint_indices, 'no_samples':no_samples}

	# 	tc.add_monitor({"name":"termination_criteria", "expression":tc.ocp.objective, "reference":0.01, "lower":True, "initial":True})
	# 	sim_type = "bullet_notrealtime"

	# 	mpc_obj = MPC.MPC(tc, sim_type, mpc_params)

	# 	#run the ocp with IPOPT to get a good initial guess for the MPC
	# 	mpc_obj.configMPC_fromcurrent()

	# 	#run the MPC
	# 	mpc_obj.runMPC()


	# 	obj.run_simulation(1000)
	# 	obj.end_simulation()
