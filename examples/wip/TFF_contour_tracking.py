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

if __name__ == "__main__":

	print("Task specification and visualization of MPC of force contouring task")
	bullet_mpc_nr = True #visualize and simulate the task as MPC

	#some task settings
	horizon_size = 1
	t_mpc = 0.02 #the MPC sampling time
	max_joint_vel = 30*3.14159/180
	max_joint_acc = 30*3.14159/180

	#load the robot and obtain the states and controls for acceleration-limited MPC
	robot = rob.Robot('iiwa7')
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	tc = tp.task_context(horizon_size*t_mpc)
	q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	q0 = [0]*7 #a configuration where KUKA is vertical
	#a pose where the EE is facing downwards
	q0 = [ 1.99590045e-05,  2.84250916e-01, -2.48833090e-05, -1.53393297e+00, 6.94133172e-06,  1.32338963e+00, -5.61481048e-06]
	#joint pose where EE is facing forwards
	q0 = [-7.07045204e-08, -2.00641310e-01,  6.11552435e-08, -1.85148630e+00, 1.18393300e-08, -8.01683098e-02, -3.72281394e-08]

	offset = 0.02 #in the z direction

	if bullet_mpc_nr:

		from tasho import world_simulator
		from tasho import MPC
		from tasho import utils
		
		#from utils import geometry

		bullet_world = world_simulator.world_simulator()

		#create a different set of objects for different force control contour tracing tasks
		#Spawns an object in the bullet environment and also the contour path (hence the task frame) 
		#is determined in the frame of the robot base.
		task_index = 0
		if task_index == 0:
			#contour tracing on a planar surface where the curvature of the contour to be traced is zero
			bullet_world.add_cylinder(0.5, 0.5, 500, {'position':[0.7, 0.0, 0.25], 'orientation':[0., 0., 0., 1.0]})
			def contour_path(s):
				y = -0.25 + 0.5*s
				x = 0.5
				z = 0.5
				return cs.vertcat(x, y, z)
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

		#EE term
		fk_vals = robot.fk(q)[7]
		p_des = contour_path(s)

		contour_error = {'lub':True, 'hard': True, 'expression':fk_vals[0:3,3] - p_des, 'upper_limits':[0.005]*3, 'lower_limits':[-0.005]*3}
		vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':0.1}
		s_regularization = {'hard': False, 'expression':s, 'reference':1..1, 'gain':0.1, 'norm':'L1'} #push towards contour tracing
		s_dot_regularization = {'hard': False, 'expression':s_dot, 'reference':0.0, 'gain':0.01, 'norm':'L2'}
		s_ddot_regularization = {'hard': False, 'expression':s_ddot, 'reference':0, 'gain':0.1}
		s_con = {'hard':True, 'lub':True, 'expression':s, 'upper_limits':1.0, 'lower_limits':0}
		s_dotcon = {'hard':True, 'lub':True, 'expression':s_dot, 'upper_limits':3, 'lower_limits':0}
		# task_objective = {'path_constraints':[vel_regularization, s_dot_regularization, s_con]}
		task_objective = {'path_constraints':[contour_error,  vel_regularization, s_regularization, s_dot_regularization, s_con, s_dotcon, s_ddot_regularization]}

		#set all joint velocities to zero
		bullet_world.resetJointState(kukaID, joint_indices, q0)
		bullet_world.setController(kukaID, "velocity", joint_indices, targetVelocities = [0]*7)
		bullet_world.enableJointForceSensor(kukaID, [6])
		bullet_world.run_simulation(1000)
		jointState = bullet_world.readJointState(kukaID, [6])
		print("Direct output of the force sensor")
		print(jointState[0][2])
		#filter the force sensor reading to compensate for the mass of the last link to estimate the
		#force applied at the end effector in the global reference frame

		#7th ([6]) element of the output of robot.fk() provides the frame corresponding to the 7th joint
		#which is the same joint where the torque sensor is enabled.
		print(robot.fk(q0)[6])
		jointPose = np.array(robot.fk(q0)[6])
		invJointPose = utils.geometry.inv_T_matrix(jointPose) #compute the inverse of the transformation
		#	matrix of the joint pose
		print(invJointPose)
		reactionGravVector = np.array([0, 0, 9.81])
		mass_last_link = 0.3 #kg
		#compute the reaction force experienced by the joint due to the mass of the last link
		force_last_link = cs.mtimes(invJointPose[0:3, 0:3], reactionGravVector)*mass_last_link

		#compute the corrected joint reaction force
		joint_force_original = np.array(jointState[0][2], ndmin = 2).T
		joint_force_corrected = joint_force_original
		joint_force_corrected[0:3] = joint_force_corrected[0:3] - force_last_link
		print("Compensated joint reaction force")
		print(joint_force_corrected)

		bullet_world.end_simulation()