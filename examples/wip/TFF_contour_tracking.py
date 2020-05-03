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


	if bullet_mpc_nr:

		from tasho import world_simulator
		from tasho import MPC

		bullet_world = world_simulator.world_simulator()

		position = [0., 0., 0.]
		orientation = [0., 0., 0., 1.]
		kukaID = bullet_world.add_robot(position, orientation, 'iiwa7')
		no_samples = int(t_mpc / bullet_world.physics_ts)

		if no_samples != t_mpc / bullet_world.physics_ts:
			print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

		joint_indices = [0, 1, 2, 3, 4, 5, 6]

		#set all joint velocities to zero
		bullet_world.setController(kukaID, "velocity", joint_indices, targetVelocities = [0]*7)
		bullet_world.enableJointForceSensor(kukaID, [6])
		bullet_world.run_simulation(1000)
		jointState = bullet_world.readJointState(kukaID, [6])
		print(jointState)
		bullet_world.end_simulation()


