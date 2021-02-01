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

    print("Random bin picking with Kinova Gen3")

    visualizationBullet = True
    horizon_size = 20
    t_mpc = 0.1
    max_joint_acc = 60 * 3.14159 / 180

    robot = rob.Robot("kinova")
    robot.set_joint_acceleration_limits(lb=-max_joint_acc, ub=max_joint_acc)
    print(robot.joint_name)
    print(robot.joint_ub)
    print(robot.joint_lb)

    # --------------------------------------------------------------------------
    # Approximation to object
    # --------------------------------------------------------------------------
    tc = tp.task_context(horizon_size * t_mpc)

    q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

    # computing the expression for the final frame
    print(robot.fk)
    fk_vals = robot.fk(q)[7]
    print(fk_vals[0:2, 3])

    # T_goal = np.array([[0.0, 0., -1., 0.5], [0., 1., 0., 0.], [1.0, 0., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
    # T_goal = np.array([[0., 0., -1., 0.5], [-1., 0., 0., 0.], [0., 1., 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
    # T_goal = np.array([[0., 1., 0., 0.5], [1., 0., 0., 0.], [0., 0., -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
    cube_pos = tc.create_expression("cube_pos", 'parameter', (3,1))
    T_goal = cs.vertcat(cs.hcat([0, 1, 0, cube_pos[0]]), cs.hcat([1, 0, 0, cube_pos[1]]), cs.hcat([0, 0, -1, cube_pos[2]]), cs.hcat([0, 0, 0, 1]))
    # T_goal = np.array([[0, 1, 0, 0], [1, 0, 0, -0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]])
    tc.ocp.set_value(cube_pos, [0.5, 0, 0.25])
    final_pos = {
        "hard": False,
        "type": "Frame",
        "expression": fk_vals,
        "reference": T_goal, 'rot_gain':10, 'trans_gain':10, 'norm':'L1'
    }
    final_vel = {"hard": True, "expression": q_dot, "reference": 0}
    final_constraints = {"final_constraints": [final_pos, final_vel]}
    tc.add_task_constraint(final_constraints)

    # adding penality terms on joint velocity and position
    vel_regularization = {"hard": False, "expression": q_dot, "reference": 0, "gain": 1e-3}
    acc_regularization = {
        "hard": False,
        "expression": q_ddot,
        "reference": 0,
        "gain": 1e-3,
    }

    task_objective = {"path_constraints": [vel_regularization, acc_regularization]}
    tc.add_task_constraint(task_objective)

    tc.set_ocp_solver("ipopt")
    # tc.set_ocp_solver(
    #     "ipopt",
    #     {
    #         "ipopt": {
    #             "max_iter": 1000,
    #             "hessian_approximation": "limited-memory",
    #             "limited_memory_max_history": 5,
    #             "tol": 1e-3,
    #         }
    #     },
    # )
    # q0_val = [0]*7
    # q0_val = [0, -3.1416/6, 0, 3*3.1416/6, 0, 3.1416/6, 0]
    # q0_val = [0, 3.1416/6, 0, 4*3.1416/6, 0, -2*3.1416/6, -1.5708]
    # q0_val = [0, 2.8942, 1.0627, 1.4351,-0.7789, 1.4801, -3.0595]
    # q0_val = [0, -0.523598, 3.141592654, -2.61799, 3.141592654, -0.523598, -1.5708]
    q0_val = [0, -0.523598, 0, 2.51799, 0, -0.523598, -1.5708]
    # q0_val = [0, 0.3491, 0, 2.0944, 0, 0.6981, -1.5708]

    tc.ocp.set_value(q0, q0_val)
    tc.ocp.set_value(q_dot0, [0] * 7)
    disc_settings = {
        "discretization method": "multiple shooting",
        "horizon size": horizon_size,
        "order": 1,
        "integration": "rk",
    }
    tc.set_discretization_settings(disc_settings)
    sol = tc.solve_ocp()

    ts, q_sol = sol.sample(q, grid="control")
    print(q_sol)
    print(robot.fk(q_sol[-1, :])[7])
    final_qsol_approx = q_sol[-1, :]
    print(final_qsol_approx)

    # --------------------------------------------------------------------------
    # Demo simulation
    # --------------------------------------------------------------------------

    if visualizationBullet:

        from tasho import world_simulator
        import pybullet as p

        obj = world_simulator.world_simulator()

        position = [0.0, 0.0, 0.0]
        orientation = [0.0, 0.0, 0.0, 1.0]

        kinovaID = obj.add_robot(position, orientation, "kinova")
        # Add a cylinder to the world
        # cylID = obj.add_cylinder(0.15, 0.5, 0.5, {'position':[0.5, 0.0, 0.25], 'orientation':[0.0, 0.0, 0.0, 1.0]})
        cylID = p.loadURDF(
            "models/objects/cube_small.urdf",
            [0.5, -0.2, 0.35],
            [0.0, 0.0, 0.0, 1.0],
            globalScaling=1.0,
        )

        p.resetBaseVelocity(cylID, linearVelocity= [0,0.5,0])

        tbStartOrientation = p.getQuaternionFromEuler([0, 0, 1.5708])
        tbID = p.loadURDF(
            "models/objects/table.urdf",
            [0.5, 0, 0],
            tbStartOrientation,
            globalScaling=0.3,
        )

        no_samples = int(t_mpc / obj.physics_ts)

        if no_samples != t_mpc / obj.physics_ts:
            print(
                "[ERROR] MPC sampling time not integer multiple of physics sampling time"
            )

        # correspondence between joint numbers in bullet and OCP determined after reading joint info of YUMI
        # from the world simulator
        joint_indices = [0, 1, 2, 3, 4, 5, 6]

        # begin the visualization of applying OCP solution in open loop
        ts, q_sol = sol.sample(q, grid="control")
        ts, q_dot_sol = sol.sample(q_dot, grid="control")
        obj.resetJointState(kinovaID, joint_indices, q0_val)
        obj.setController(
            kinovaID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
        )
        #obj.run_simulation(100)  # Here, the robot is just waiting to start the task

        for i in range(horizon_size * 100):
            # Update control signal
            tc.ocp.set_value(q0, q_sol[1])
            tc.ocp.set_value(q_dot0, q_dot_sol[1])
            sol = tc.solve_ocp()
            ts, q_sol = sol.sample(q, grid="control")
            ts, q_dot_sol = sol.sample(q_dot, grid="control")
            tc.ocp.set_initial(q, q_sol.T)
            tc.ocp.set_initial(q_dot, q_dot_sol.T)

            # Set control signal
            obj.setController(
                kinovaID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
            )

            # Simulate
            obj.run_simulation(no_samples)

        obj.run_simulation(100)  # Here, the robot is just waiting to start the task

        obj.end_simulation()
