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
	max_joint_acc = 180*3.14159/180
	max_joint_vel = 90*3.14159/180
	robot = rob.Robot('iiwa7')	

	
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)

	#obj.run_simulation(1000)
	print(robot.joint_acc_lb)
	horizon_size = 5
	t_mpc = 0.1


	tc = tp.task_context(horizon_size*t_mpc)

	q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	fk_vals = robot.fk(q)[6]

	T_goal = np.array([[-1., 0., 0., -0.24], [0., 1., 0., -0.37], [0.0, 0., -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]])

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
	tc.ocp.set_value(q0, q0_val)
	tc.ocp.set_value(q_dot0, [0]*7)

	tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
	disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	tc.set_discretization_settings(disc_settings)
	sol = tc.solve_ocp()

	joint_indices = [0, 1, 2, 3, 4, 5, 6]


		#begin the visualization of applying OCP solution in open loop
	_, q_sol = sol.sample(q, grid="control")
	print(robot.fk(q_sol[-1])[6])
	print(q_sol[-1])
	ts, q_dot_sol = sol.sample(q_dot, grid="control")
	obj.resetJointState(kukaID, joint_indices, q0_val)
	obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[0])
	



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
	obj.end_simulation()
