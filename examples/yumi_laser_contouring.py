import sys
from tasho import task_prototype_rockit as tp
from tasho import robot as rob
from rockit import MultipleShooting, Ocp
import matplotlib.pyplot as plt

if __name__ == '__main__':

	print("Task specification and visualization of MPC control of laser contouring task")

	horizon_size = 10
	t_mpc = 0.1 #the MPC sampling time
	max_joint_vel = 30*3.14159/180
	max_joint_acc = 30*3.14159/180

	#TODO: remove below line after pinocchio starts to provide the robot joint limits
	rob_settings = {'n_dof' : 18, 'no_links' : 20, 'q_min' : [-2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, 0, 0, -2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, 0, 0], 'q_max' : [2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025, 2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025] }
	robot = rob.Robot('yumi')
	robot.set_joint_limits(lb = rob_settings['q_min'], ub = rob_settings['q_max'])
	print(robot.joint_ub)

	tc = tp.task_context(horizon_size*t_mpc)

	q = tc.create_expression('q', 'state', (robot.ndof, 1)) #joint positions over the trajectory
	q_dot = tc.create_expression('q_dot', 'state', (robot.ndof, 1)) #joint velocities
	q_ddot = tc.create_expression('q_ddot', 'control', (robot.ndof, 1))

	tc.set_dynamics(q, q_dot)
	tc.set_dynamics(q_dot, q_ddot)

	pos_limits = {'lub':True, 'hard': True, 'expression':q, 'upper_limits':robot.joint_ub, 'lower_limits':robot.joint_lb}
	vel_limits = {'lub':True, 'hard': True, 'expression':q_dot, 'upper_limits':max_joint_vel, 'lower_limits':-max_joint_vel}
	acc_limits = {'lub':True, 'hard': True, 'expression':q_ddot, 'upper_limits':max_joint_acc, 'lower_limits':-max_joint_acc}
	joint_constraints = {'path_constraints':[pos_limits, vel_limits, acc_limits]}
	tc.add_task_constraint(joint_constraints)

	#parameters of the ocp
	q0 = tc.create_expression('q0', 'parameter', (robot.ndof, 1))
	q_dot0 = tc.create_expression('q_dot0', 'parameter', (robot.ndof, 1))

	