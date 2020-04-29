********
Examples
********

..
    .. raw:: html
        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
            <iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
        </div>

Picking example
===============

Two tasks: *Move towards the object*, *Pick it and place it somewhere else*::

    from tasho import robot as rob
    from tasho import problem_template as pt
    from tasho import input_resolution
    from tasho import discrete_plan as dp

    import casadi as cs
    from casadi import pi, cos, sin
    from rockit import MultipleShooting, Ocp
    import numpy as np

    if __name__ == '__main__':

      ...

      # -------------------------------------
      # Instantiate discrete plan
      # -------------------------------------
        plan = dp.DiscretePlan()

      # -------------------------------------
      # Setting a robot
      # -------------------------------------
        robot = rob.Robot('kinova')
        robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
        robot.set_state(q0_val + q_dot0_val)
        robot.set_robot_input_resolution("acceleration")

      # -------------------------------------
      # Task 1: Move towards the object
      # -------------------------------------

        # Set horizon size and goal
        horizon_size = 10
        T_goal = np.array([[0, 1, 0, 0.5], [1, 0, 0, 0], [0, 0, -1, 0.25], [0, 0, 0, 1]])  # T_goal = np.array([[0.0, 0., -1., 0.5], [0., 1., 0., 0.], [1.0, 0., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]]) # T_goal = np.array([[0., 0., -1., 0.5], [-1., 0., 0., 0.], [0., 1., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]]) # T_goal = np.array([[0., 1., 0., 0.5], [1., 0., 0., 0.], [0., 0., -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]]) # T_goal = np.array([[0, 1, 0, 0], [1, 0, 0, -0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]])

        # Set task
        approach_task = pt.Point2Point(horizon_size*t_mpc, horizon = horizon_size, goal = T_goal)

        # Add robot to task
        approach_task.add_robot(robot)

      # -------------------------------------
      # Task 2: Pick and place the object
      # -------------------------------------

        horizon_size = 16
        T_goal = np.array([[1, 0, 0, 0], [0, -1, 0, -0.5], [0, 0, -1, 0.25], [0, 0, 0, 1]])

        pickup_task = pt.Point2Point(time = horizon_size*t_mpc, horizon = horizon_size, goal = T_goal)

        pickup_task.add_robot(robot)

      # --------------------------------------------------------------------------
      # Define discrete plan
      # --------------------------------------------------------------------------
        plan.add_task(approach_task, "approach")
        plan.add_task(pickup_task, "pickup")

      # --------------------------------------------------------------------------
      # Simulate plan execution
      # --------------------------------------------------------------------------
        # Get initial conditions from robot
        robot_q0 = robot.get_initial_conditions[0:robot.ndof]
        robot_qdot0 = robot.get_initial_conditions[robot.ndof:2*robot.ndof]

        # Simulate
        plan.simulate_plan(simulator = "bullet", q_init = robot_q0, qdot_init = robot_qdot0)
