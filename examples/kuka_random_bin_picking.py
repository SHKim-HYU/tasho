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

	print("Random bin picking with KUKA Iiwa")

	visualizationBullet = False
	horizon_size = 10
	t_mpc = 0.5
	max_joint_acc = 30*3.14159/180

	robot = rob.Robot('iiwa7')
	robot.set_from_json('iiwa7.json')
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	print(robot.joint_name)
	print(robot.joint_ub)
	tc = tp.task_context(horizon_size*t_mpc)

	q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	#computing the expression for the final frame
	print(robot.fk)
	fk_vals = robot.fk(q)[7]

	T_goal = np.array([[0.0, 0., -1., 0.5], [0., 1., 0., 0.], [1.0, 0., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
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
		p.createConstraint(kukaID, 6, cylID, -1, p.JOINT_FIXED, [0., 0., 1.], [0., 0, 0.1], [0., 0., 0.1])
		obj.run_simulation(1000)
		obj.end_simulation()
