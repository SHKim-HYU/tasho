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

	print("Random bin picking with KUKA Iiwa")

	visualizationBullet = True
	horizon_size = 10
	t_mpc = 0.5
	max_joint_acc = 30*3.14159/180
	max_joint_vel = 30*3.14159/180

	robot = rob.Robot('iiwa7')
	
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	print(robot.joint_name)
	print(robot.joint_ub)
	tc = tp.task_context(horizon_size*t_mpc)

	q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	#computing the expression for the final frame
	fk_vals = robot.fk(q)[6]
	# fk_vals[0:3,3] = fk_vals[0:3,3] + fk_vals[0:3, 0:3]@np.array([0.0, 0.0, 0.17])

	T_goal = np.array([[-1., 0., 0., 0.6], [0., 1., 0.0, -0.2], [0.0, 0., -1.0, 0.25], [0.0, 0.0, 0.0, 1.0]])
	final_pos = {'hard':True, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal}
	final_vel = {'hard':True, 'expression':q_dot, 'reference':0}
	final_constraints = {'final_constraints':[final_pos, final_vel]}
	tc.add_task_constraint(final_constraints)

	#adding penality terms on joint velocity and position
	vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':1}
	acc_regularization = {'hard': False, 'expression':q_ddot, 'reference':0, 'gain':1}

	task_objective = {'path_constraints':[vel_regularization, acc_regularization]}
	tc.add_task_constraint(task_objective)

	tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
	q0_val = [0]*7
	q0_val = [-0.23081576,  0.90408998,  0.02868817, -1.20917942, -0.03413408,  1.05074694, -0.19664998]
	#Next two values are initial points used for Yudha's dual arm task
	q0_val = [-5.53636820e-01, 1.86726808e-01, -1.32319806e-01, -2.06761360e+00, 3.12421835e-02,  8.89043596e-01, -7.03329152e-01]
	q0_val = [ 0.36148756, 0.19562711, 0.34339407,-2.06759027, -0.08427634, 0.89133467, 0.75131025]
	tc.ocp.set_initial(q, q0_val)
	tc.ocp.set_value(q0, q0_val)
	tc.ocp.set_value(q_dot0, [0]*7)
	disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	tc.set_discretization_settings(disc_settings)
	sol = tc.solve_ocp()

	ts, q_sol = sol.sample(q, grid="control")
	print(q_sol)
	print(robot.fk(q_sol[-1,:])[7])


	if visualizationBullet:

		from tasho import world_simulator
		import pybullet as p

		obj = world_simulator.world_simulator()

		position = [0.0, 0.0, 0.0]
		orientation = [0.0, 0.0, 0.0, 1.0]

		kukaID = obj.add_robot(position, orientation, 'iiwa7')
		#Add a cylinder to the world
		#cylID = obj.add_cylinder(0.15, 0.5, 0.5, {'position':[0.5, 0.0, 0.25], 'orientation':[0.0, 0.0, 0.0, 1.0]})
		cylID = p.loadURDF("cube_small.urdf", [0.5, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 3.0)
		#print(obj.getJointInfoArray(kukaID))
		no_samples = int(t_mpc/obj.physics_ts)

		if no_samples != t_mpc/obj.physics_ts:
			print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

		#correspondence between joint numbers in bullet and OCP determined after reading joint info of YUMI
		#from the world simulator
		joint_indices = [0, 1, 2, 3, 4, 5, 6]

		#begin the visualization of applying OCP solution in open loop
		ts, q_dot_sol = sol.sample(q_dot, grid="control")
		obj.resetJointState(kukaID, joint_indices, q0_val)
		obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[0])
		obj.run_simulation(480)


		for i in range(horizon_size):
			q_vel_current = 0.5*(q_dot_sol[i] + q_dot_sol[i+1])
			obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_vel_current)
			obj.run_simulation(no_samples)

		obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol[-1])


		#create a constraint to attach the body to the robot (TODO: make this systematic and add to the world_simulator class)
		#print(p.getNumBodies())
		#get link state of EE
		ee_state = p.getLinkState(kukaID, 6, computeForwardKinematics = True)
		print(ee_state)
		#get link state of the cylinder
		#cyl_state = p.getLinkState(cylID, -1, computeForwardKinematics = True)
		#print(cyl_state)
		#p.createConstraint(kukaID, 6, cylID, -1, p.JOINT_FIXED, [0., 0., 1.], [0., 0, 0.1], [0., 0., 0.1])

		#testing the collision detection times of pybullet
		boxIDs = []
		boxIDs.append(cylID)

		cylID2 = p.loadURDF("cube_small.urdf", [0.6, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID3 = p.loadURDF("cube_small.urdf", [0.4, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID4 = p.loadURDF("cube_small.urdf", [0.6, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID5 = p.loadURDF("cube_small.urdf", [0.6, 0.1, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID6 = p.loadURDF("cube_small.urdf", [0.6, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID7 = p.loadURDF("cube_small.urdf", [0.4, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID8 = p.loadURDF("cube_small.urdf", [0.6, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID9 = p.loadURDF("cube_small.urdf", [0.6, 0.1, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID10 = p.loadURDF("cube_small.urdf", [0.6, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID11 = p.loadURDF("cube_small.urdf", [0.4, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID12 = p.loadURDF("cube_small.urdf", [0.6, 0, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		cylID13 = p.loadURDF("cube_small.urdf", [0.6, 0.1, 0.25], [0.0, 0.0, 0.0, 1.0], globalScaling = 2.0)
		obj.run_simulation(100)
		boxIDs = [cylID, cylID2, cylID3, cylID4, cylID5, cylID6, cylID7, cylID8, cylID9, cylID10, cylID11, cylID12, cylID13]
		tic = time.time()
		for i in range(1000):
			for cyl_id in boxIDs:
				coll_shape = p.getClosestPoints(kukaID, cyl_id, 10.0)
		print(time.time() - tic)

		obj.run_simulation(1000)
		obj.end_simulation()
