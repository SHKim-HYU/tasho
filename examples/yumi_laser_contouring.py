import sys
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np
import matplotlib.pyplot as plt

# Yumi laser tracing OCP with visualization in bullet

def inv_T_matrix(T):

	T_inv = cs.horzcat(cs.horzcat(T[0:3, 0:3].T, cs.mtimes(-T[0:3, 0:3].T, T[0:3,3])).T, [0, 0, 0, 1]).T

	return T_inv

def circle_path(s, centre, radius, rot_mat = np.eye(3)):

	T_goal = cs.vertcat(centre[0] + radius*(cos(s)-1), centre[1] + radius*sin(s), centre[2])
	
	return T_goal

if __name__ == '__main__':

	print("Task specification and visualization of MPC control of laser contouring task")

	visualizationBullet = False #toggle visualization with PyBullet option

	horizon_size = 20
	t_mpc = 0.2 #the MPC sampling time
	max_joint_vel = 30*3.14159/180
	max_joint_acc = 30*3.14159/180

	#TODO: remove below line after pinocchio starts to provide the robot joint limits
	rob_settings = {'n_dof' : 18, 'no_links' : 20, 'q_min' : np.array([-2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, 0, 0, -2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, 0, 0]).T, 'q_max' : np.array([2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025, 2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025]).T }
	robot = rob.Robot('yumi')
	robot.set_from_json('yumi.json')
	print(robot.joint_name)
	## Customise robot limits
	# robot.set_joint_limits(lb = rob_settings['q_min'], ub = rob_settings['q_max'])
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)

	print(robot.joint_ub)
	print(robot.joint_ub)

	tc = tp.task_context(horizon_size*t_mpc)

	q = tc.create_expression('q', 'state', (robot.ndof, 1)) #joint positions over the trajectory
	q_dot = tc.create_expression('q_dot', 'state', (robot.ndof, 1)) #joint velocities
	q_ddot = tc.create_expression('q_ddot', 'control', (robot.ndof, 1))

	s = tc.create_expression('s', 'state', (1, 1)) #Progress variable for the contour tracing task
	s_dot = tc.create_expression('s_dot', 'control', (1, 1)) 

	tc.set_dynamics(q, q_dot)
	tc.set_dynamics(q_dot, q_ddot)
	tc.set_dynamics(s, s_dot)


	#TODO: change to using limtis from robot object!
	pos_limits = {'lub':True, 'hard': True, 'expression':q, 'upper_limits':rob_settings['q_max'], 'lower_limits':rob_settings['q_min']}
	vel_limits = {'lub':True, 'hard': True, 'expression':q_dot, 'upper_limits':robot.joint_vel_ub, 'lower_limits':robot.joint_vel_lb}
	acc_limits = {'lub':True, 'hard': True, 'expression':q_ddot, 'upper_limits':robot.joint_acc_ub, 'lower_limits':robot.joint_acc_lb}
	joint_constraints = {'path_constraints':[pos_limits, vel_limits, acc_limits]}
	# joint_constraints = {'path_constraints':[pos_limits]}
	tc.add_task_constraint(joint_constraints)

	#parameters of the ocp
	q0 = tc.create_expression('q0', 'parameter', (robot.ndof, 1))
	q_dot0 = tc.create_expression('q_dot0', 'parameter', (robot.ndof, 1))

	#adding the initial constraints on joint position and velocity
	joint_init_con = {'expression':q, 'reference':q0}
	joint_vel_init_con = {'expression':q_dot, 'reference':q_dot0}
	s_init_con = {'expression':s, 'reference':0}
	s_dot_init_con = {'expression':s_dot, 'reference':0}
	init_constraints = {'initial_constraints':[joint_init_con, joint_vel_init_con, s_init_con, s_dot_init_con]}
	# init_constraints = {'initial_constraints':[joint_init_con]}
	tc.add_task_constraint(init_constraints)

	#computing the forward kinematics of the robot tree
	fk_vals = robot.fk(q)

	fk_left_ee = fk_vals[7]
	fk_right_ee = fk_vals[17]

	#computing the point of projection of laser on the plane
	fk_relative = cs.mtimes(inv_T_matrix(fk_right_ee), fk_left_ee)
	Ns = np.array([0, 0, 1], ndmin=2).T
	sc = -0.3
	Nl = fk_relative[0:3,2]
	P = fk_relative[0:3,3]
	a = (-sc - cs.mtimes(Ns.T, P))/cs.mtimes(Ns.T, Nl)
	Ps = P + a*Nl

	# #representing the contour profile as a function of the progress variable
	centre = [0.0, 0.0, -sc]
	radius = 0.1
	p_des = circle_path(s, centre, radius)

	#contour_error = {'equality':True, 'hard': False, 'expression':Ps, 'reference':p_des, 'gain':10}
	#contour_error = {'equality':True, 'hard': True, 'expression':Ps, 'reference':p_des, 'gain':10}
	contour_error = {'lub':True, 'hard': True, 'expression':Ps - p_des, 'upper_limits':[0.002]*3, 'lower_limits':[-0.002]*3}
	vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':0.1}
	s_dot_regularization = {'hard': False, 'expression':s_dot, 'reference':0, 'gain':0.01}
	s_con = {'hard':True, 'inequality':True, 'expression':-s, 'upper_limits':0}
	# task_objective = {'path_constraints':[vel_regularization, s_dot_regularization, s_con]}
	task_objective = {'path_constraints':[contour_error, vel_regularization, s_dot_regularization, s_con]}
	

	#Add path constraints on the depth and the angle of the laser interception
	dot_prod_ee_workpiece = -cs.mtimes(Ns.T, Nl)
	#constraint on the angle between laser pointer and workpiece normal
	angle_limit = 10*3.14159/180
	angle_constraint = {'hard':True, 'inequality':True, 'expression':-dot_prod_ee_workpiece, 'upper_limits':angle_limit}
	#constraint on the distance between laser pointer and the workpiece
	distance_constraint = {'hard':True, 'lub':True, 'expression':a, 'upper_limits':0.02, 'lower_limits':0.0}
	task_objective['path_constraints'].append(angle_constraint)
	task_objective['path_constraints'].append(distance_constraint)

	tc.add_task_constraint(task_objective)

	#adding the final constraints
	final_vel = {'hard':True, 'expression':q_dot, 'reference':0}
	final_s = {'hard':True, 'expression':s, 'reference':6.28}
	final_constraints = {'final_constraints':[final_vel, final_s]}
	tc.add_task_constraint(final_constraints)
	q0_contour = np.array([-1.35488912e+00, -8.72846052e-01, 2.18411843e+00,  6.78786296e-01,
  2.08696971e+00, -9.76390128e-01, -1.71721329e+00,  1.65969745e-03,
  1.65969745e-03,  1.47829337e+00, -5.24943547e-01, -1.95134781e+00,
  5.30517837e-01, -2.69960026e+00, -8.14070355e-01,  1.17172289e+00,
  2.06459136e-03,  2.06462524e-03]).T

	tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
	# tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'tol':1e-3}})
	tc.ocp.set_value(q0, q0_contour)
	tc.ocp.set_value(q_dot0, [0]*18)
	disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	tc.set_discretization_settings(disc_settings)
	try:
		sol = tc.solve_ocp()
		#print(sol.sample(q, grid="control"))
	except:
		tc.ocp.show_infeasibilities(1e-3)
		sol = tc.ocp.debug
		sol = tc.ocp.opti.debug
		# print(sol.value(tc.ocp._method.eval_at_control(tc.ocp, q, 0)))

	print(sol.sample(Ps - p_des, grid="control"))

	if visualizationBullet:

		from tasho import world_simulator

		obj = world_simulator.world_simulator()

		position = [0.0, 0.0, 0.0]
		orientation = [0.0, 0.0, 0.0, 1.0]

		yumiID = obj.add_robot(position, orientation, 'yumi')
		no_samples = int(t_mpc/obj.physics_ts)

		if no_samples != t_mpc/obj.physics_ts:
			print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

		#correspondence between joint numbers in bullet and OCP determined after reading joint info of YUMI
		#from the world simulator
		joint_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

		#begin the visualization of applying OCP solution in open loop
		ts, q_dot_sol = sol.sample(q_dot, grid="control")
		obj.resetJointState(yumiID, joint_indices, q0_contour)
		obj.setController(yumiID, "velocity", joint_indices, targetVelocities = q_dot_sol[0])
		obj.run_simulation(480)


		for i in range(horizon_size):
			q_vel_current = 0.5*(q_dot_sol[i] + q_dot_sol[i+1])
			obj.setController(yumiID, "velocity", joint_indices, targetVelocities = q_vel_current)
			obj.run_simulation(no_samples)

		obj.setController(yumiID, "velocity", joint_indices, targetVelocities = q_dot_sol[0])
		obj.run_simulation(1000)
		obj.end_simulation()
