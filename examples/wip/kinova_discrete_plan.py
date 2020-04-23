import sys
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np
import matplotlib.pyplot as plt
from tasho import discrete_plan as dp

if __name__ == '__main__':

	print("Random bin picking with Kinova Gen3")

	visualizationBullet = True
	horizon_size = 10
	t_mpc = 0.5
	max_joint_acc = 30*3.14159/180

	plan = dp.DiscretePlan()

	robot = rob.Robot('kinova')
	robot.set_from_json('kinova.json')
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	print(robot.joint_name)
	print(robot.joint_ub)
	print(robot.joint_lb)

	# --------------------------------------------------------------------------
	# Approximation to object
	# --------------------------------------------------------------------------
	tc = tp.task_context(horizon_size*t_mpc)

	q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

	#computing the expression for the final frame
	print(robot.fk)
	fk_vals = robot.fk(q)[7]
	print(fk_vals[0:2,3])

	# T_goal = np.array([[0.0, 0., -1., 0.5], [0., 1., 0., 0.], [1.0, 0., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
	# T_goal = np.array([[0., 0., -1., 0.5], [-1., 0., 0., 0.], [0., 1., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
	# T_goal = np.array([[0., 1., 0., 0.5], [1., 0., 0., 0.], [0., 0., -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
	T_goal = np.array([[0, 1, 0, 0.5], [1, 0, 0, 0], [0, 0, -1, 0.25], [0, 0, 0, 1]])
	# T_goal = np.array([[0, 1, 0, 0], [1, 0, 0, -0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]])
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
	# q0_val = [0]*7
	# q0_val = [0, -3.1416/6, 0, 3*3.1416/6, 0, 3.1416/6, 0]
	# q0_val = [0, 3.1416/6, 0, 4*3.1416/6, 0, -2*3.1416/6, -1.5708]
	# q0_val = [0, 2.8942, 1.0627, 1.4351,-0.7789, 1.4801, -3.0595]
	# q0_val = [0, -0.523598, 3.141592654, -2.61799, 3.141592654, -0.523598, -1.5708]
	q0_val = [0, -0.523598, 0, 2.51799, 0, -0.523598, -1.5708]
	# q0_val = [0, 0.3491, 0, 2.0944, 0, 0.6981, -1.5708]


	tc.ocp.set_value(q0, q0_val)
	tc.ocp.set_value(q_dot0, [0]*7)
	disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	tc.set_discretization_settings(disc_settings)

	plan.add_task(tc)
	plan.add_task(tc)
	# plan.add_task(tc,"p2p")
	# plan.add_task(tc)
	# plan.add_task(tc)


	plan.print_tasks()

	sol_list = plan.execute_plan()
	print(sol_list)

	plan.simulate_plan()

	# sol = tc.solve_ocp()

	# ts, q_sol = sol.sample(q, grid="control")
	# print(q_sol)
	# print(robot.fk(q_sol[-1,:])[7])
	# final_qsol_approx = q_sol[-1,:]
	# print(final_qsol_approx)

	# # --------------------------------------------------------------------------
	# # Pick the object up
	# # --------------------------------------------------------------------------
	# horizon_size_pickup = 16
    #
	# tc_pickup = tp.task_context(horizon_size_pickup*t_mpc)
    #
	# q_pickup, q_dot_pickup, q_ddot_pickup, q0_pickup, q_dot0_pickup = input_resolution.acceleration_resolved(tc_pickup, robot, {})
    #
	# #computing the expression for the final frame
	# print(robot.fk)
	# fk_vals = robot.fk(q_pickup)[7]
    #
	# T_goal_pickup = np.array([[1, 0, 0, 0], [0, -1, 0, -0.5], [0, 0, -1, 0.25], [0, 0, 0, 1]])
	# # T_goal_pickup = np.array([[0, 1, 0, 0], [1, 0, 0, -0.5], [0, 0, -1, 0.25], [0, 0, 0, 1]])
	# final_pos_pickup = {'hard':True, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal_pickup}
	# final_vel_pickup = {'hard':True, 'expression':q_dot_pickup, 'reference':0}
	# final_constraints_pickup = {'final_constraints':[final_pos_pickup, final_vel_pickup]}
	# tc_pickup.add_task_constraint(final_constraints_pickup)
    #
	# #adding penality terms on joint velocity and position
	# vel_regularization = {'hard': False, 'expression':q_dot_pickup, 'reference':0, 'gain':1}
	# acc_regularization = {'hard': False, 'expression':q_ddot_pickup, 'reference':0, 'gain':1}
    #
	# task_objective_pickup = {'path_constraints':[vel_regularization, acc_regularization]}
	# tc_pickup.add_task_constraint(task_objective_pickup)
    #
	# tc_pickup.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
	# q0_val_pickup = final_qsol_approx
	# tc_pickup.ocp.set_value(q0_pickup, q0_val_pickup)
	# tc_pickup.ocp.set_value(q_dot0_pickup, [0]*7)
	# disc_settings_pickup = {'discretization method': 'multiple shooting', 'horizon size': horizon_size_pickup, 'order':1, 'integration':'rk'}
	# tc_pickup.set_discretization_settings(disc_settings_pickup)
	# sol_pickup = tc_pickup.solve_ocp()
    #
	# ts_pickup, q_sol_pickup = sol_pickup.sample(q_pickup, grid="control")
