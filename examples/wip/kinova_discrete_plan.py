
from tasho import robot as rob
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import discrete_plan as dp

import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np

if __name__ == '__main__':

    print("Random bin picking with Kinova Gen3")

    horizon_size = 10
    t_mpc = 0.5
    max_joint_acc = 30*3.14159/180

    q0_val = [0, -0.523598, 0, 2.51799, 0, -0.523598, -1.5708]

    plan = dp.DiscretePlan()

    robot = rob.Robot('kinova')

    robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
    print(robot.joint_name)
    print(robot.joint_ub)
    print(robot.joint_lb)

    # --------------------------------------------------------------------------
    # Approximation to object
    # --------------------------------------------------------------------------
    tc = tp.task_context(horizon_size*t_mpc)

    # q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})
    q, q_dot, q_ddot, q0, q_dot0 = robot.set_input_resolution(tc,"acceleration")
    # q = robot.states["q"]
    # q_dot = robot.states["q_dot"]
    # q_ddot = robot.inputs["q_ddot"]
    # q0 = robot.parameters["q0"]
    # q_dot0 = robot.parameters["q_dot0"]

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

    disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
    tc.set_discretization_settings(disc_settings)

    # --------------------------------------------------------------------------
    # Picking the object up
    # --------------------------------------------------------------------------

    horizon_size_pickup = 16

    tc_pickup = tp.task_context(horizon_size_pickup*t_mpc)

    q_pickup, q_dot_pickup, q_ddot_pickup, q0_pickup, q_dot0_pickup = input_resolution.acceleration_resolved(tc_pickup, robot, {})

    #computing the expression for the final frame
    print(robot.fk)
    fk_vals = robot.fk(q_pickup)[7]

    T_goal_pickup = np.array([[1, 0, 0, 0], [0, -1, 0, -0.5], [0, 0, -1, 0.25], [0, 0, 0, 1]])
    # T_goal_pickup = np.array([[0, 1, 0, 0], [1, 0, 0, -0.5], [0, 0, -1, 0.25], [0, 0, 0, 1]])
    final_pos_pickup = {'hard':True, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal_pickup}
    final_vel_pickup = {'hard':True, 'expression':q_dot_pickup, 'reference':0}
    final_constraints_pickup = {'final_constraints':[final_pos_pickup, final_vel_pickup]}
    tc_pickup.add_task_constraint(final_constraints_pickup)

    #adding penality terms on joint velocity and position
    vel_regularization = {'hard': False, 'expression':q_dot_pickup, 'reference':0, 'gain':1}
    acc_regularization = {'hard': False, 'expression':q_ddot_pickup, 'reference':0, 'gain':1}

    task_objective_pickup = {'path_constraints':[vel_regularization, acc_regularization]}
    tc_pickup.add_task_constraint(task_objective_pickup)

    tc_pickup.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
    disc_settings_pickup = {'discretization method': 'multiple shooting', 'horizon size': horizon_size_pickup, 'order':1, 'integration':'rk'}
    tc_pickup.set_discretization_settings(disc_settings_pickup)


    # --------------------------------------------------------------------------
    # Define discrete plan
    # --------------------------------------------------------------------------
    plan.add_task(tc, "approach")
    plan.add_task(tc_pickup, "pickup")

    plan.print_tasks()

    # sol = plan.solve_task(task_name = "approach", q_init = q0_val)
    # sol_list = plan.solve_task(task_name = ["approach","pickup"], q_init = q0_val)
    # sol_list = plan.execute_plan(q_init = q0_val)

    # --------------------------------------------------------------------------
    # Simulate plan execution
    # --------------------------------------------------------------------------

    plan.simulate_plan(simulator = "bullet", q_init = q0_val)
