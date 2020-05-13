#Example for contour tracking with force control and utilizing Task Frame Formalism

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

#TODO: add rk2 discretization in rockit, because do not need rk4 for constant acceleration.

if __name__ == "__main__":

	print("Task specification and visualization of MPC of force contouring task")
	bullet_mpc_nr = True #visualize and simulate the task as MPC

	#some task settings
	horizon_size = 10
	t_mpc = 0.02 #the MPC sampling time
	max_joint_vel = 60*3.14159/180
	max_joint_acc = 60*3.14159/180

	#load the robot and obtain the states and controls for acceleration-limited MPC
	robot = rob.Robot('iiwa7')
	#print(robot.id([0]*7, [0]*7, [0]*7))
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	jac_fun = robot.set_kinematic_jacobian('kin_jac', 6)
	tc = tp.task_context(horizon_size*t_mpc)
	q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	print(robot.joint_ub)
	print(robot.joint_lb)

	q1 = [0]*7 #a configuration where KUKA is vertical
	#a pose where the EE is facing downwards
	q1 = [ 1.99590045e-05,  2.84250916e-01, -2.48833090e-05, -1.53393297e+00, 6.94133172e-06,  1.32338963e+00, -5.61481048e-06]
	#joint pose where EE is facing forwards
	q1 = [-7.07045204e-08, -2.00641310e-01,  6.11552435e-08, -1.85148630e+00, 1.18393300e-08, -8.01683098e-02, -3.72281394e-08]

	offset = 0.02 #in the z direction
	print(jac_fun(q1))
	if bullet_mpc_nr:

		from tasho import world_simulator
		from tasho import MPC
		from tasho import utils
		import pybullet as p
		
		#from utils import geometry

		bullet_world = world_simulator.world_simulator()

		#create a different set of objects for different force control contour tracing tasks
		#Spawns an object in the bullet environment and also the contour path (hence the task frame) 
		#is determined in the frame of the robot base.
		task_index = 0
		if task_index == 0:
			#contour tracing on a planar surface where the curvature of the contour to be traced is zero
			bullet_world.add_cube({'position':[0.7, 0.0, 0.25], 'orientation':[0., 0., 0., 1.0]}, scale = 0.55)
			def contour_path(s):
				y = -0.25 + 0.5*s
				x = 0.5
				z = 0.545
				return cs.vertcat( cs.horzcat(cs.MX.eye(3),cs.vertcat(x, y, z)), cs.MX([0, 0, 0, 1]).T)
			q1 = [-3.97670686e-01,  4.30757898e-01, -9.47372587e-02, -1.33677556e+00,
   3.99392326e-02,  1.37526019e+00, -4.91678720e-01]
		elif task_index == 1:
			#contour tracing on the curved surface of a cylinder. So the curvature is constant
			print("Not implemented")
		elif task_index == 2:
			#contour tracing on a suface with changing curvature
			print("Not implemented")

		#Adding the KUKA robot
		position = [0., 0., 0.]
		orientation = [0., 0., 0., 1.]
		kukaID = bullet_world.add_robot(position, orientation, 'iiwa7')
		no_samples = int(t_mpc / bullet_world.physics_ts)

		if no_samples != t_mpc / bullet_world.physics_ts:
			print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

		joint_indices = [0, 1, 2, 3, 4, 5, 6]

		#Begininning the task specification
		#progress variables
		s = tc.create_expression('s', 'state', (1, 1)) #Progress variable for the contour tracing task
		s_dot = tc.create_expression('s_dot', 'state', (1, 1))
		s_ddot = tc.create_expression('s_ddot', 'control', (1, 1))
		tc.set_dynamics(s, s_dot)
		tc.set_dynamics(s_dot, s_ddot)
		s0 = tc.create_expression('s0', 'parameter', (1, 1))
		s_dot0 = tc.create_expression('s_dot0', 'parameter', (1, 1))
		s_init_con = {'expression':s, 'reference':s0}
		s_dot_init_con = {'expression':s_dot, 'reference':s_dot0}
		init_constraints = {'initial_constraints':[s_init_con, s_dot_init_con]}
		tc.add_task_constraint(init_constraints)

		force_desired = tc.create_expression('f_des', 'parameter', (3, 1))
		force_measured = tc.create_expression('f_meas', 'parameter', (3,1))
		#q_dot_force = tc.create_expression('q_dot_force', 'variable', (7,1))
		K = 0.05*0 #proportional gain of the feedback force controller
		jac_val = jac_fun(q0)
		q_dot_force = cs.mtimes(jac_val.T, cs.solve(cs.mtimes(jac_val, jac_val.T) + 1e-6, K*(force_desired - force_measured)))
		q_dot_force_fun = cs.Function('q_dot_force_fun', [q0, force_desired, force_measured], [q_dot_force])

		# tc.ocp.set_value(force_desired, [0,0,0])
		# tc.ocp.set_value(force_measured, [0,0,0])

		#EE term
		fk_vals = robot.fk(q)[6]
		p_des = contour_path(s)

		contour_error_soft = {'hard': False, 'expression':fk_vals[0:3,3], 'reference':p_des[0:3, 3], 'gain':5.0, 'norm':'L2'}
		contour_error = {'lub':True, 'hard': True, 'expression':fk_vals[0:3,3] - p_des[0:3, 3], 'upper_limits':[0.005, 0.005, 0.005], 'lower_limits':[-0.005, -0.005, -0.1]}
		vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':0.1}
		s_regularization = {'hard': False, 'expression':s, 'reference':1.1, 'gain':0.5*0, 'norm':'L1'} #push towards contour tracing
		#s_dot_regularization = {'hard': False, 'expression':s_dot, 'reference':0.3, 'gain':0.1, 'norm':'L2'}
		s_dot_regularization = {'hard': False, 'expression':s_dot, 'reference':0.3, 'gain':1.0, 'norm':'L1'}
		s_ddot_regularization = {'hard': False, 'expression':s_ddot, 'reference':0, 'gain':0.1}
		s_con = {'hard':True, 'lub':True, 'expression':s, 'upper_limits':1.0, 'lower_limits':0}
		s_dotcon = {'hard':True, 'lub':True, 'expression':s_dot, 'upper_limits':3, 'lower_limits':0}
		#q_dot_force_con = {'hard':True, 'expression':q_dot_force, 'reference':cs.mtimes(jac_val.T, cs.solve(cs.mtimes(jac_val, jac_val.T) + 1e-4, K*(force_desired - force_measured)))}
		task_objective = {'path_constraints':[contour_error_soft, vel_regularization, s_regularization, s_ddot_regularization, s_dotcon,  s_dot_regularization, s_con]}
		#task_objective = {'path_constraints':[contour_error,  vel_regularization, s_regularization, s_dot_regularization, s_con, s_dotcon, s_ddot_regularization]}

		tc.add_task_constraint(task_objective)
		
		#first attempt at the force control. Do motion planning only for the motion part and for the force part,
		#use a feedback control loop to apply the force.
		#need the direction to apply force, and the P -term
		#set all joint velocities to zero
		bullet_world.resetJointState(kukaID, joint_indices, q1)
		bullet_world.setController(kukaID, "velocity", joint_indices, targetVelocities = [0]*7)

		#unlock the torque control mode
		# p.setJointMotorControlArray(kukaID, [0,1,2,3,4,5,6], p.VELOCITY_CONTROL, forces=[0]*7)
		# for link_idx in joint_indices:
		# 	p.changeDynamics(kukaID, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
		#check if the torque control works with a high rate of control
		# for i in range(100):
		# 	jointsInfo = bullet_world.readJointState(kukaID, joint_indices)
		# 	q1 = []
		# 	for jointInfo in jointsInfo:
		# 		q1.append(jointInfo[0])
		# 	torques = p.calculateInverseDynamics(kukaID, q1, [0]*7, [0.1]*7)	
		# 	for j in range(4):
		# 		bullet_world.setController(kukaID, "torque", [0,1,2,3,4,5,6], targetTorques = torques)
		# 		p.stepSimulation()
		# 		time.sleep(bullet_world.physics_ts)

		bullet_world.enableJointForceSensor(kukaID, [6])
		#bullet_world.run_simulation(100)
		jointState = bullet_world.readJointState(kukaID, [6])
		print("Direct output of the force sensor")
		print(jointState[0][2])

		tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 100, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
		# tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'tol':1e-3}})
		tc.ocp.set_value(q0, q1)
		tc.ocp.set_value(q_dot0, [0]*7)
		tc.ocp.set_value(s0, 0)
		tc.ocp.set_value(s_dot0, 0)
		#tc.ocp.set_initial(s_dot, )
		disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
		tc.set_discretization_settings(disc_settings)
		#sol = tc.solve_ocp()
		try:
			sol = tc.solve_ocp()
			_, s_sol = sol.sample(s, grid = 'control')
			print(s_sol)
			#print(sol.sample(q, grid="control"))
		except:
			tc.ocp.show_infeasibilities(1e-3)
			sol = tc.ocp.debug
			sol = tc.ocp.opti.debug
			# print(sol.value(tc.ocp._method.eval_at_control(tc.ocp, q, 0)))

		tc.add_monitor({"name":"termination_criteria", "expression":s, "reference":0.99, "greater":True, "initial":True})

		#configuring the parameters of the MPC
		mpc_params = {'world':bullet_world}
		q0_params_info = {'type':'joint_position', 'joint_indices':joint_indices, 'robotID':kukaID}
		q_dot0_params_info = {'type':'joint_velocity', 'joint_indices':joint_indices, 'robotID':kukaID}
		s0_params_info = {'type':'progress_variable', 'state':True}
		s_dot0_params_info = {'type':'progress_variable', 'state':True}
		mpc_params['params'] = {'q0':q0_params_info, 'q_dot0':q_dot0_params_info, 's0':s0_params_info, 's_dot0':s_dot0_params_info, 'robots':{kukaID:robot}}
		mpc_params['params']['f_des'] = {'type':'set_value', 'value':np.array([0,0,-200])}

		#creating a function to pass as a parameter to the MPC class to appropriately post process 
		#the sensor readings
		def joint_force_compensation(fk, q, force):
			mass_last_link = 0.3
			reactionGravVector = np.array([0, 0, 9.81])
			jointPose = np.array(fk(q)[6])
			invJointPose = utils.geometry.inv_T_matrix(jointPose)
			force_last_link = cs.mtimes(invJointPose[0:3, 0:3], reactionGravVector)*mass_last_link
			force_corrected = force - force_last_link
			force_corrected = cs.mtimes(jointPose[0:3, 0:3], force_corrected)
			return force_corrected

		mpc_params['params']['f_meas'] = {'type':'joint_force', 'robotID':kukaID, 'joint_indices':[6], 'fk':robot.fk, 'post_process':joint_force_compensation}
		mpc_params['disc_settings'] = disc_settings
		# mpc_params['solver_name'] = 'ipopt'
		# mpc_params['solver_params'] = {'lbfgs':True}
		mpc_params['solver_name'] = 'sqpmethod'
		mpc_params['solver_params'] = {'ipopt':True}
		mpc_params['t_mpc'] = t_mpc
		mpc_params['control_type'] = 'joint_acceleration' #'joint_velocity'
		mpc_params['control_info'] = {'force_control':True, 'fcon_fun':q_dot_force_fun, 'robotID':kukaID, 'discretization':'constant_acceleration', 'joint_indices':joint_indices, 'no_samples':no_samples}
		# set the joint positions in the simulator
		bullet_world.resetJointState(kukaID, joint_indices, q1)
		sim_type = "bullet_notrealtime"
		mpc_obj = MPC.MPC(tc, sim_type, mpc_params)

		#run the ocp with IPOPT to get a good initial guess for the MPC
		mpc_obj.configMPC_fromcurrent()

		#run the MPC
		mpc_obj.runMPC()

		#bullet_world.run_simulation(1000)

		#filter the force sensor reading to compensate for the mass of the last link to estimate the
		#force applied at the end effector in the global reference frame

		#7th ([6]) element of the output of robot.fk() provides the frame corresponding to the 7th joint
		#which is the same joint where the torque sensor is enabled.
		# print(q1)


		print(robot.fk(q1)[6])
		jointPose = np.array(robot.fk(q1)[6])
		invJointPose = utils.geometry.inv_T_matrix(jointPose) #compute the inverse of the transformation
		#	matrix of the joint pose
		print(invJointPose)
		reactionGravVector = np.array([0, 0, 9.81])
		mass_last_link = 0.3 #kg
		#compute the reaction force experienced by the joint due to the mass of the last link
		force_last_link = cs.mtimes(invJointPose[0:3, 0:3], reactionGravVector)*mass_last_link

		#compute the corrected joint reaction force
		joint_force_original = np.array(jointState[0][2], ndmin = 2).T
		print(joint_force_original)
		joint_force_corrected = joint_force_original
		joint_force_corrected[0:3] = joint_force_corrected[0:3] - force_last_link
		print("Compensated joint reaction force")
		print(joint_force_original)

		bullet_world.end_simulation()