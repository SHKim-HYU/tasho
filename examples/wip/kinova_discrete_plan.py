
from tasho import robot as rob
from tasho import discrete_plan as dp

from tasho import problem_template as pt

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
    q_dot0_val = [0]*7

    # Instantiate plan
    plan = dp.DiscretePlan()

    # Set robot
    robot = rob.Robot('kinova')
    robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
    robot.set_state(q0_val + q_dot0_val)
    robot.set_robot_input_resolution("acceleration")

    # # --------------------------------------------------------------------------
    # # Approximation to object
    # # --------------------------------------------------------------------------
    horizon_size = 10
    T_goal = np.array([[0, 1, 0, 0.5], [1, 0, 0, 0], [0, 0, -1, 0.25], [0, 0, 0, 1]])  # T_goal = np.array([[0.0, 0., -1., 0.5], [0., 1., 0., 0.], [1.0, 0., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]]) # T_goal = np.array([[0., 0., -1., 0.5], [-1., 0., 0., 0.], [0., 1., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]]) # T_goal = np.array([[0., 1., 0., 0.5], [1., 0., 0., 0.], [0., 0., -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]]) # T_goal = np.array([[0, 1, 0, 0], [1, 0, 0, -0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]])

    approach_task = pt.Point2Point(horizon_size*t_mpc, horizon = horizon_size, goal = T_goal)
    # approach_task = tp.Point2Point(horizon = horizon_size, goal = T_goal)
    approach_task.add_robot(robot)

    # --------------------------------------------------------------------------
    # Picking the object up
    # --------------------------------------------------------------------------
    horizon_size = 16
    T_goal = np.array([[1, 0, 0, 0], [0, -1, 0, -0.5], [0, 0, -1, 0.25], [0, 0, 0, 1]])

    pickup_task = pt.Point2Point(time = horizon_size*t_mpc, horizon = horizon_size, goal = T_goal)
    # pickup_task = tp.Point2Point(horizon = horizon_size, goal = T_goal)
    pickup_task.add_robot(robot)


    # --------------------------------------------------------------------------
    # Define discrete plan
    # --------------------------------------------------------------------------
    plan.add_task(approach_task, "approach")
    plan.add_task(pickup_task, "pickup")

    plan.print_tasks()

    # sol = plan.solve_task(task_name = "approach", q_init = q0_val)
    # sol_list = plan.solve_task(task_name = ["approach","pickup"], q_init = q0_val)
    # sol_list = plan.execute_plan(q_init = q0_val)

    # --------------------------------------------------------------------------
    # Simulate plan execution
    # --------------------------------------------------------------------------
    robot_q0 = robot.get_initial_conditions[0:robot.ndof]
    robot_qdot0 = robot.get_initial_conditions[robot.ndof:2*robot.ndof]

    plan.simulate_plan(simulator = "bullet", q_init = robot_q0, qdot_init = robot_qdot0)
