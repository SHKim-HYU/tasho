## OCP for point-to-point motion and visualization of a KUKA robot arm

from tasho import task_prototype_rockit as tp
from tasho import input_resolution, world_simulator
from tasho import robot as rob
from tasho import MPC
from tasho.utils import geometry
from casadi import pi, cos, sin, acos
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

if __name__ == "__main__":

    print("Task specification and visualization of P2P OCP")

    horizon_size = 16
    t_mpc = 0.02

    robot_choice = "kinova"
    ocp_control = "torque_resolved"  #'acceleration_resolved' #"torque_resolved"  #

    robot = rob.Robot(robot_choice)
    if ocp_control == 'acceleration_resolved':
        max_joint_acc = 240*3.14159/180
        robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)

    print(robot.joint_name)
    print(robot.joint_ub)
    print(robot.joint_vel_ub)
    print(robot.joint_acc_ub)
    print(robot.joint_torque_ub)

    q_init = [0, pi / 6, 0, 4 * pi / 6, 0, -2 * pi / 6, -pi / 2]

    ##########################################
    # Define contour
    ##########################################
    def contour_path(s):
        ee_fk_init = robot.fk(q_init)[7]
        ee_pos_init = ee_fk_init[:3, 3]
        ee_rot_init = ee_fk_init[:3, :3]

        sdotref = 0.2
        sdot_path = sdotref * (
            5.777783e-13 * s ** 5
            - 34.6153846154 * s ** 4
            + 69.2307692308 * s ** 3
            - 46.7307692308 * s ** 2
            + 12.1153846154 * s
            + 0.0515384615
        )

        a_p = 0.15
        z_p = 0.05
        pos_path = ee_pos_init + cs.vertcat(
            z_p * sin(s * (4 * pi)),
            a_p * sin(s * (2 * pi)),
            a_p * sin(s * (2 * pi)) * cos(s * (2 * pi)),
        )
        rot_path = ee_rot_init

        return pos_path, rot_path, sdot_path

    ##########################################
    # Task spacification
    ##########################################
    tc = tp.task_context(horizon_size * t_mpc)

    if ocp_control == "acceleration_resolved":
        q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(
            tc, robot, {}
        )
    elif ocp_control == "torque_resolved":
        q, q_dot, q_ddot, tau, q0, q_dot0 = input_resolution.torque_resolved(
            tc, robot, {"forward_dynamics_constraints": False}
        )

    # Define augmented dynamics based on path-progress variable s
    s = tc.create_expression("s", "state", (1, 1))
    s_dot = tc.create_expression("s_dot", "state", (1, 1))
    s_ddot = tc.create_expression("s_ddot", "control", (1, 1))

    tc.set_dynamics(s, s_dot)
    tc.set_dynamics(s_dot, s_ddot)

    pos_path, rot_path, sdot_path = contour_path(s)

    # Set s(0) and s_dot(0) as parameters
    s0 = tc.create_expression("s0", "parameter", (1, 1))
    s_dot0 = tc.create_expression("s_dot0", "parameter", (1, 1))

    s_init_con = {"expression": s, "reference": s0}
    s_dot_init_con = {"expression": s_dot, "reference": s_dot0}
    init_constraints = {"initial_constraints": [s_init_con, s_dot_init_con]}
    tc.add_task_constraint(init_constraints)

    s_con = {
        "hard": True,
        "lub": True,
        "expression": s,
        "lower_limits": 0,
        "upper_limits": 1,
    }
    s_dotcon = {
        "hard": True,
        "inequality": True,
        "expression": -s_dot,
        "upper_limits": 0,
    }
    s_path_constraints = {"path_constraints": [s_con, s_dotcon]}
    tc.add_task_constraint(s_path_constraints)

    # Set tunnel constraints
    def pos_err(q, s):
        ee_fk = robot.fk(q)[7]
        return ee_fk[:3, 3] - pos_path

    def rot_err(q, s):
        ee_fk = robot.fk(q)[7]
        ee_rot_n = ee_fk[:3, 0]
        ee_rot_s = ee_fk[:3, 1]
        ee_rot_a = ee_fk[:3, 2]

        path_rot_n = rot_path[:, 0]
        path_rot_s = rot_path[:, 1]
        path_rot_a = rot_path[:, 2]

        return 0.5 * (
            geometry.cross_vec2vec(ee_rot_n, path_rot_n)
            + geometry.cross_vec2vec(ee_rot_s, path_rot_s)
            + geometry.cross_vec2vec(ee_rot_a, path_rot_a)
        )

    # pos_tunnel_con = cs.sumsqr(pos_err(q, s)) - rho^2 <= slack
    pos_tunnel_con = {
        "hard": False,
        "inequality": True,
        "expression": pos_err(q, s),
        "upper_limits": 0.01 ** 2,
        "gain": 100,
        "norm": "squaredL2",
    }

    tunnel_constraints = {"path_constraints": [pos_tunnel_con]}
    tc.add_task_constraint(tunnel_constraints)

    # adding penality terms on joint velocity and position
    # pos_regularization = {"hard": False, "expression": q, "reference": 0, "gain": 1e-2}
    # vel_regularization = {
    #     "hard": False,
    #     "expression": q_dot,
    #     "reference": 0,
    #     "gain": 1e-2,
    # }

    # task_objective = {"path_constraints": [pos_regularization, vel_regularization]}
    # tc.add_task_constraint(task_objective)
    tc.add_regularization(expression=(s_dot - sdot_path), weight=20, norm="L2")
    tc.add_regularization(expression=pos_err(q, s), weight=1e-1, norm="L2")
    tc.add_regularization(expression=rot_err(q, s), weight=1e-1, norm="L2")

    if ocp_control == "torque_resolved":
        tc.add_regularization(
            expression=tau, weight=4e-5, norm="L2", variable_type="control", reference=0
        )
    if ocp_control == "acceleration_resolved":
        tc.add_regularization(
            expression=q_ddot, weight=1e-3, norm="L2", variable_type="control", reference=0
        )
    tc.add_regularization(
        expression=s_ddot, weight=4e-5, norm="L2", variable_type="control", reference=0
    )

    tc.add_regularization(
        expression=q, weight=1e-2, norm="L2", variable_type="state", reference=0
    )
    tc.add_regularization(
        expression=q_dot, weight=1e-2, norm="L2", variable_type="state", reference=0
    )
    tc.add_objective(
        tc.ocp.at_tf(
            1e-5
            * cs.sumsqr(
                cs.vertcat(
                    1e-2 * q,
                    10 * q_dot,
                    1e-2 * (1 - s),
                    10 * s_dot,
                    10 * pos_err(q, s),
                    10 * rot_err(q, s),
                )
            )
        )
    )
    tc.set_ocp_solver('ipopt')
    # tc.set_ocp_solver(
    #     "ipopt",
    #     {
    #         "ipopt": {
    #             "print_level": 0,
    #             "max_iter": 1000,
    #             "hessian_approximation": "limited-memory",
    #             "limited_memory_max_history": 5,
    #             "tol": 1e-3,
    #             "dual_inf_tol": 1e-3,
    #             "compl_inf_tol": 1e-3,
    #             "constr_viol_tol": 1e-3,
    #             "acceptable_tol": 1e-3,
    #         },
    #         # "print_time": False,
    #     },
    # )

    tc.ocp.set_value(q0, q_init)
    tc.ocp.set_value(q_dot0, [0] * 7)
    tc.ocp.set_value(s0, 0)
    tc.ocp.set_value(s_dot0, 0)

    disc_settings = {
        "discretization method": "multiple shooting",
        "horizon size": horizon_size,
        "order": 1,
        "integration": "rk",
    }
    tc.set_discretization_settings(disc_settings)
    sol = tc.solve_ocp()

    # ts, q_sol = sol.sample(q, grid="control")
    # ts, q_dot_sol = sol.sample(q_dot, grid="control")
    # ts, s_sol = sol.sample(s, grid="control")
    # ts, s_dot_sol = sol.sample(s_dot, grid="control")

    # # Define solve for online execution (MPC)
    # kkt_tol_pr = 1e-5  # 1e-3
    # kkt_tol_du = 1e-5  # 1e-1
    # min_step_size = 1e-16  # 1e-6
    # max_iter = 2  # 1
    # max_iter_ls = 3  # 0
    # qpsol_options = {
    #     "constr_viol_tol": kkt_tol_pr,
    #     "dual_inf_tol": kkt_tol_du,
    #     "verbose": False,
    #     "print_iter": False,
    #     # "print_header": False,
    # }
    # solver_options = {
    #     "qpsol": "qrqp",
    #     "qpsol_options": qpsol_options,
    #     "verbose": False,
    #     "tol_pr": kkt_tol_pr,
    #     "tol_du": kkt_tol_du,
    #     # "hessian_approximation": "limited-memory",
    #     # "lbfgs_memory": 5,
    #     "min_step_size": min_step_size,
    #     "max_iter": max_iter,
    #     "max_iter_ls": max_iter_ls,
    #     "print_iteration": True,
    #     "print_header": True,
    #     "print_status": True,
    #     "print_time": True,
    # }
    # tc.set_ocp_solver("sqpmethod", solver_options)

    # tc.set_discretization_settings(disc_settings)

    # tc.ocp.set_value(q0, q_sol[1])
    # tc.ocp.set_value(q_dot0, q_dot_sol[1])
    # tc.ocp.set_value(s0, s_sol[1])
    # tc.ocp.set_value(s_dot0, s_dot_sol[1])

    # tc.ocp.set_initial(q, q_sol.T)
    # tc.ocp.set_initial(q_dot, q_dot_sol.T)
    # tc.ocp.set_initial(s, s_sol.T)
    # tc.ocp.set_initial(s_dot, s_dot_sol.T)

    # sol = tc.solve_ocp()

    # --------------------------------------------------------------------------
    # Demo simulation
    # --------------------------------------------------------------------------
    # visualizationBullet = False
    #
    # if not visualizationBullet:
    #
    #     for i in range(horizon_size * 10000):
    #         # Update robot state
    #         ts, q_sol = sol.sample(q, grid="control")
    #         ts, q_dot_sol = sol.sample(q_dot, grid="control")
    #         ts, s_sol = sol.sample(s, grid="control")
    #         ts, s_dot_sol = sol.sample(s_dot, grid="control")
    #
    #         tc.ocp.set_value(q0, q_sol[1])
    #         tc.ocp.set_value(q_dot0, q_dot_sol[1])
    #         tc.ocp.set_value(s0, s_sol[1])
    #         tc.ocp.set_value(s_dot0, s_dot_sol[1])
    #
    #         tc.ocp.set_initial(q, q_sol.T)
    #         tc.ocp.set_initial(q_dot, q_dot_sol.T)
    #         tc.ocp.set_initial(s, s_sol.T)
    #         tc.ocp.set_initial(s_dot, s_dot_sol.T)
    #
    #         print("\n------- Solution --------")
    #         print("s: ", s_sol[0])
    #         print("s_dot: ", s_dot_sol[0], "\n")
    #         ts, pos_err_sol = sol.sample(cs.sumsqr(pos_err(q, s)), grid="control")
    #         print("Position error: ", pos_err_sol[0], "\n")
    #
    #         # Solve the ocp again
    #         sol = tc.solve_ocp()
    #
    # else:
    #
    #     from tasho import world_simulator
    #     import pybullet as p
    #
    #     obj = world_simulator.world_simulator(bullet_gui=False)
    #
    #     position = [0.0, 0.0, 0.0]
    #     orientation = [0.0, 0.0, 0.0, 1.0]
    #
    #     kinovaID = obj.add_robot(position, orientation, "kinova")
    #
    #     no_samples = int(t_mpc / obj.physics_ts)
    #
    #     if no_samples != t_mpc / obj.physics_ts:
    #         print(
    #             "[ERROR] MPC sampling time not integer multiple of physics sampling time"
    #         )
    #
    #     # correspondence between joint numbers in bullet and OCP determined after reading joint info of YUMI
    #     # from the world simulator
    #     joint_indices = [0, 1, 2, 3, 4, 5, 6]
    #
    #     # begin the visualization of applying OCP solution in open loop
    #     # ts, q_sol = sol.sample(q, grid="control")
    #     # ts, q_dot_sol = sol.sample(q_dot, grid="control")
    #     obj.resetJointState(kinovaID, joint_indices, q_init)
    #     obj.setController(kinovaID, "velocity", joint_indices, targetVelocities=[0] * 7)
    #     # obj.run_simulation(100)  # Here, the robot is just waiting to start the task
    #
    #     for i in range(horizon_size * 100):
    #
    #         # Update robot state
    #         ts, q_sol = sol.sample(q, grid="control")
    #         ts, q_dot_sol = sol.sample(q_dot, grid="control")
    #         ts, s_sol = sol.sample(s, grid="control")
    #         ts, s_dot_sol = sol.sample(s_dot, grid="control")
    #
    #         tc.ocp.set_value(q0, q_sol[1])
    #         tc.ocp.set_value(q_dot0, q_dot_sol[1])
    #         tc.ocp.set_value(s0, s_sol[1])
    #         tc.ocp.set_value(s_dot0, s_dot_sol[1])
    #
    #         tc.ocp.set_initial(q, q_sol.T)
    #         tc.ocp.set_initial(q_dot, q_dot_sol.T)
    #         tc.ocp.set_initial(s, s_sol.T)
    #         tc.ocp.set_initial(s_dot, s_dot_sol.T)
    #
    #         # print("########### ---> ", s_sol[0])
    #
    #         # Solve the ocp again
    #         sol = tc.solve_ocp()
    #
    #         # Set control signal
    #         obj.setController(
    #             kinovaID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
    #         )
    #         # Simulate
    #         obj.run_simulation(no_samples)
    #
    #     obj.run_simulation(100)  # Here, the robot is just waiting to start the task
    #
    #     obj.end_simulation()

    # print(q_sol)
    # print(robot.fk(q_sol[-1, :])[7])

    use_MPC_class = True

    if use_MPC_class:

        from tasho import world_simulator
        import pybullet as p

        obj = world_simulator.world_simulator(bullet_gui=False)

        position = [0.0, 0.0, 0.0]
        orientation = [0.0, 0.0, 0.0, 1.0]

        kinovaID = obj.add_robot(position, orientation, "kinova")

        no_samples = int(t_mpc / obj.physics_ts)

        if no_samples != t_mpc / obj.physics_ts:
            print(
                "[ERROR] MPC sampling time not integer multiple of physics sampling time"
            )

        # correspondence between joint numbers in bullet and OCP determined after reading joint info of YUMI
        # from the world simulator
        joint_indices = [0, 1, 2, 3, 4, 5, 6]
        obj.resetJointState(kinovaID, joint_indices, q_init)
        obj.setController(kinovaID, "velocity", joint_indices, targetVelocities=[0] * 7)

        mpc_params = {'world':obj}
        q0_params_info = {'type':'joint_position', 'joint_indices':joint_indices, 'robotID':kinovaID}
        q_dot0_params_info = {'type':'joint_velocity', 'joint_indices':joint_indices, 'robotID':kinovaID}
        s0_params_info = {'type':'progress_variable', 'state':True}
        s_dot0_params_info = {'type':'progress_variable', 'state':True}
        mpc_params['params'] = {'q0':q0_params_info, 'q_dot0':q_dot0_params_info, 's0':s0_params_info, 's_dot0':s_dot0_params_info, 'robots':{kinovaID:robot}}
        mpc_params['disc_settings'] = disc_settings
        # mpc_params['solver_name'] = 'ipopt'
        # mpc_params['solver_params'] = {'lbfgs':True}
        mpc_params['solver_name'] = 'sqpmethod'
        mpc_params['solver_params'] = {'qrqp':True}
        mpc_params['t_mpc'] = t_mpc
        mpc_params['control_type'] = 'joint_velocity' #'joint_torque'
        mpc_params['control_info'] = {'robotID':kinovaID, 'discretization':'constant_acceleration', 'joint_indices':joint_indices, 'no_samples':no_samples}
        # set the joint positions in the simulator
        sim_type = "bullet_notrealtime"
        tc.add_monitor({"name":"termination_criteria", "expression":s, "reference":0.98, "greater":True, "initial":True})
        mpc_obj = MPC.MPC(tc, sim_type, mpc_params)
        mpc_obj.max_mpc_iter = 400
        #run the ocp with IPOPT to get a good initial guess for the MPC
        mpc_obj.configMPC_fromcurrent()
        mpc_obj.runMPC()
