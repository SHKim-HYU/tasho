#code to benchmark the benefits of a predictive horizon even for a very simple
#point-to-point motion task

import sys
from tasho import task_prototype_rockit as tp
from tasho import input_resolution, world_simulator, MPC
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pickle

if __name__ == '__main__':


	robot = rob.Robot('iiwa7')
	max_joint_acc = 180*3.14159/180
	max_joint_vel = 90*3.14159/180
	position = [0.0, 0.0, 0.0]
	orientation = [0.0, 0.0, 0.0, 1.0]
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)

	#create a library of joint values to be stored
	# q_lib = [[0.6967786678678314, 1.0571249256028108, 0.14148034853277666, -1.270205899164967, 0.24666659678004457, 0.7847437220601475, 0.41090241207031053],
	# [0]*7]
	# T_goal_lib = [np.array([[-1., 0., 0., -0.24], [0., 1., 0., -0.37], [0.0, 0., -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]]),
	# 0]

	

	#Uncomment to generate a set of random configurations
	#Code to generate a library of joint configurations to benchmark
	#the performance of the MPC
	# q_source_array = []
	# q_destination_array = []

	# for i in range(50):
	# 	q_source_array.append(robot.generate_random_configuration())
	# 	q_destination_array.append(robot.generate_random_configuration())

	# array_library = {'q_source_array':q_source_array, 'q_destination_array':q_destination_array}
	# with open('examples/wip/tasho_paper_examples/p2p_array_library.txt', 'w') as filehandle:
	# 	json.dump(array_library, filehandle)

	#code to load the json q_source and destination arrays
	with open('examples/wip/tasho_paper_examples/p2p_array_library.txt', 'r') as fp:
		array_library = json.load(fp)

	q_source_array = array_library['q_source_array']
	q_destination_array = array_library['q_destination_array']

	
	results = {}

	horizon_sizes = [2,3,4,5,6,7,8,9,10,12,14,16,18,20, 24, 28, 32, 36, 40, 50]
	# horizon_sizes = [5,10]
	for i in range(50):
		for horizon_size in horizon_sizes:
			#obj.run_simulation(1000)
			print(robot.joint_acc_lb)
			# horizon_size = 40
			t_mpc = 0.1

			obj = world_simulator.world_simulator()

			kukaID = obj.add_robot(position, orientation, 'iiwa7')
		

			tc = tp.task_context(horizon_size*t_mpc)
			q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})
			fk_vals = robot.fk(q)[6]

			# T_goal = np.array([[-1., 0., 0., -0.24], [0., 1., 0., -0.37], [0.0, 0., -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
			rand_conf = robot.generate_random_configuration()
			print("Random goal pose at the configuration")
			print(rand_conf)
			T_goal = robot.fk(rand_conf)[6].full()
			T_goal = robot.fk(q_destination_array[i])[6].full()

			# final_pos = {'hard':True, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal}
			final_pos = {'hard':False, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal, 'rot_gain':1, 'trans_gain':10, 'norm':'L1'}
			# final_pos = {'hard':False, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal, 'rot_gain':1, 'trans_gain':10, 'norm':'L2'}
	
			# final_vel = {'hard':True, 'expression':q_dot, 'reference':0}
			# final_constraints = {'final_constraints':[final_pos]}
			# final_constraints = {'final_constraints':[final_vel]}
			# tc.add_task_constraint(final_constraints)

			vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':0.1}
			acc_regularization = {'hard': False, 'expression':q_ddot, 'reference':0, 'gain':0.1}
			task_objective = {'path_constraints':[final_pos, vel_regularization, acc_regularization]}
			tc.add_task_constraint(task_objective)

			q0_val = [0]*7
			q0_val = [-0.23081576,  0.90408998,  0.02868817, -1.20917942, -0.03413408,  1.05074694, -0.19664998]
			# q0_val = [0.6967786678678314, 1.0571249256028108, 0.14148034853277666, -1.270205899164967, 0.24666659678004457, 0.7847437220601475, 0.41090241207031053]
			# q0_val = robot.generate_random_configuration()
			q0_val = q_source_array[i]
			# tc.ocp.set_value(q0, q0_val)
			# tc.ocp.set_value(q_dot0, [0]*7)

			# tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
			disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
			# tc.set_discretization_settings(disc_settings)
			# sol = tc.solve_ocp()

			joint_indices = [0, 1, 2, 3, 4, 5, 6]


			#begin the visualization of applying OCP solution in open loop
			# _, q_sol = sol.sample(q, grid="control")
			# print(robot.fk(q_sol[-1])[6])
			# print(q_sol[-1])
			# ts, q_dot_sol = sol.sample(q_dot, grid="control")
			obj.resetJointState(kukaID, joint_indices, q0_val)
			# obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[0])
	



			# no_samples = int(t_mpc/obj.physics_ts)
			# for i in range(horizon_size):
			# 	q_vel_current = 0.5*(q_dot_sol[i] + q_dot_sol[i+1])
			# 	obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_vel_current)
			# 	obj.run_simulation(no_samples)

			# obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[-1])
			# obj.run_simulation(100)

			no_samples = int(t_mpc/obj.physics_ts)
			mpc_params = {'world':obj}
			q0_params_info = {'type':'joint_position', 'joint_indices':joint_indices, 'robotID':kukaID}
			q_dot0_params_info = {'type':'joint_velocity', 'joint_indices':joint_indices, 'robotID':kukaID}
			mpc_params['params'] = {'q0':q0_params_info, 'q_dot0':q_dot0_params_info, 'robots':{kukaID:robot}}

			mpc_params['disc_settings'] = disc_settings
			# mpc_params['solver_name'] = 'ipopt'
			# mpc_params['solver_params'] = {'lbfgs':True}
			mpc_params['solver_name'] = 'sqpmethod'
			# mpc_params['solver_params'] = {'qpoases':True}
			# mpc_params['solver_params'] = {'qrqp':True}
			# mpc_params['solver_params'] = {'osqp':True}
			mpc_params['solver_params'] = {'ipopt':True}
			mpc_params['t_mpc'] = t_mpc
			mpc_params['control_type'] = 'joint_acceleration' #'joint_velocity' #
			mpc_params['control_info'] = { 'robotID':kukaID, 'discretization':'constant_acceleration', 'joint_indices':joint_indices, 'no_samples':no_samples}

			goal_error = cs.vec(T_goal) - cs.vec(robot.fk(q)[6])
			tc.add_monitor({"name":"termination_criteria", "expression":cs.sumsqr(goal_error) + cs.sumsqr(q_dot), "reference":0.05, "lower":True, "initial":True})
			sim_type = "bullet_notrealtime"

			mpc_obj = MPC.MPC(tc, sim_type, mpc_params)

			#run the ocp with IPOPT to get a good initial guess for the MPC
			mpc_obj.configMPC_fromcurrent()

			# 	#run the MPC
			status = mpc_obj.runMPC()
			print(status)
			print("Time taken by the solver in each iterations")
			print(mpc_obj._solver_time)
			print("The integral of the torque^2 of effort")
			print(mpc_obj.torque_effort_sumsqr)
			solver_time = mpc_obj._solver_time
			results[str(horizon_size) +',' + str(i)] = {'status':status, 'torque_effort':list(mpc_obj.torque_effort_sumsqr.full()[0]), 'avg_solver_time':sum(solver_time)/len(solver_time), 'total_trajectory_time':t_mpc*len(solver_time)}
			obj.end_simulation()

	print(results)
	with open('examples/wip/tasho_paper_examples/p2p_results.txt', 'w') as fp:
		json.dump(results, fp)
