import unittest
from tasho import task_prototype_rockit as tp
import casadi as cs


class TestTask(unittest.TestCase):
    def test_add_regularization(self):
        tc = tp.task_context(2)

        x = tc.create_expression("x", "state", (1, 1))
        u = tc.create_expression("u", "control", (1, 1))
        tc.set_dynamics(x, u)

        task_spec = {"initial_constraints": [{"expression": x, "reference": 0.05}]}
        tc.add_task_constraint(task_spec)

        tc.add_regularization(x, 10, reference=5)
        tc.add_regularization(u, 1)

        solver_options = {
            "ipopt": {"print_level": 0},
            "print_time": False,
            "expand": False,
        }
        tc.set_ocp_solver("ipopt", solver_options)
        disc_settings = {
            "discretization method": "multiple shooting",
            "horizon size": 10,
            "order": 2,
            "integration": "rk",
        }
        tc.set_discretization_settings(disc_settings)

        sol = tc.solve_ocp()
        t, x_val = tc.sol_sample(x, grid="control")

        # test the result with L2 regularization
        self.assertAlmostEqual(
            x_val[-1], 4.971907742047628, 5, "Final position test failed"
        )
        self.assertEqual(t[-1], 2, "Final time test failed")

        # test the result with high L1 regularization on control.  no motion
        tc.add_regularization(u, 200, norm="L1")
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        t, x_val = tc.sol_sample(x, grid="control")
        self.assertAlmostEqual(x_val[-1], 0.05, 6, "Final position test failed")

        # Testing regularizations on variables
        y = tc.create_expression("y", "variable", (1, 1))
        tc.add_regularization(y, 10, variable_type="variable", reference=5)
        tc.add_regularization(y, 5, variable_type="variable", reference=0)
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = tc.sol_value(y)
        self.assertAlmostEqual(y_val, 3.333333333, 6, "Variable regularization failed")

        tc.add_regularization(y, 5, variable_type="variable", reference=0)
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = tc.sol_value(y)
        self.assertAlmostEqual(
            y_val, 2.500000000000, 6, "Variable regularization failed"
        )

        tc.add_regularization(y, 101, variable_type="variable", reference=0, norm="L1")
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = tc.sol_value(y)
        self.assertAlmostEqual(y_val, 0, 6, "Variable regularization failed")

    def test_initial_constraints(self):
        # Testing the addition of the initial constraints
        solver_options = {
            "ipopt": {"print_level": 0},
            "print_time": False,
            "expand": False,
        }
        tc = tp.task_context(2)

        x = tc.create_expression("x", "state", (1, 1))
        u = tc.create_expression("u", "control", (1, 1))
        tc.set_dynamics(x, u)

        tc.add_task_constraint(
            {"initial_constraints": [{"expression": x, "reference": 0.05}]}
        )
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": u, "reference": 1.0}]}
        )
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 10,
                "order": 1,
                "integration": "rk",
            }
        )
        sol = tc.solve_ocp()
        _, x_val = tc.sol_sample(x, grid="control")
        _, u_val = tc.sol_sample(u, grid="control")
        self.assertAlmostEqual(
            x_val[0], 0.05, 6, "initial constraint on scalar state not respected"
        )
        self.assertAlmostEqual(
            u_val[0], 1.0, 6, "initial constraint on scalar control not respected"
        )

        del tc

        tc = tp.task_context(2)

        x = tc.create_expression("x", "state", (2, 1))
        u = tc.create_expression("u", "control", (2, 1))
        tc.set_dynamics(x, u)

        tc.add_task_constraint(
            {"initial_constraints": [{"expression": x, "reference": [0.05, 0.1]}]}
        )
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": u, "reference": [1.0, -1.0]}]}
        )
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 10,
                "order": 1,
                "integration": "rk",
            }
        )
        sol = tc.solve_ocp()
        _, x_val = tc.sol_sample(x, grid="control")
        _, u_val = tc.sol_sample(u, grid="control")
        self.assertAlmostEqual(
            x_val[0, 0], 0.05, 6, "initial constraint on vector state not respected"
        )
        self.assertAlmostEqual(
            x_val[0, 1], 0.1, 6, "initial constraint on vector state not respected"
        )
        self.assertAlmostEqual(
            u_val[0, 0], 1.0, 6, "initial constraint on vector control not respected"
        )
        self.assertAlmostEqual(
            u_val[0, 1], -1.0, 6, "initial constraint on vector control not respected"
        )

    def test_final_constraints(self):
        # Testing the addition of the final constraints
        solver_options = {
            "ipopt": {"print_level": 0},
            "print_time": False,
            "expand": False,
        }
        # testing hard terminal constraints on scalars
        tc = tp.task_context(2)
        x = tc.create_expression("x", "state", (1, 1))
        u = tc.create_expression("u", "control", (1, 1))
        tc.set_dynamics(x, u)

        tc.add_task_constraint(
            {"final_constraints": [{"expression": x, "reference": 0.05, "hard": True}]}
        )
        tc.add_task_constraint(
            {"final_constraints": [{"expression": u, "reference": 1.0, "hard": True}]}
        )
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 10,
                "order": 1,
                "integration": "rk",
            }
        )
        sol = tc.solve_ocp()
        _, x_val = tc.sol_sample(x, grid="control")
        _, u_val = tc.sol_sample(u, grid="control")
        self.assertAlmostEqual(
            x_val[-1], 0.05, 6, "terminal constraint on scalar state not respected"
        )
        self.assertAlmostEqual(
            u_val[-1], 1.0, 6, "terminal constraint on scalar control not respected"
        )

        del tc

        # testing hard terminal constraints on vectors
        tc = tp.task_context(2)

        x = tc.create_expression("x", "state", (2, 1))
        u = tc.create_expression("u", "control", (2, 1))
        tc.set_dynamics(x, u)
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": x, "reference": [0.0, 0.0]}]}
        )
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": u, "reference": [0.0, 0.0]}]}
        )
        tc.add_regularization(x, 10)
        tc.add_regularization(u, 10)

        tc.add_task_constraint(
            {
                "final_constraints": [
                    {"expression": x, "reference": [0.05, 0.1], "hard": True}
                ]
            }
        )
        tc.add_task_constraint(
            {
                "final_constraints": [
                    {"expression": u, "reference": [1.0, -1.0], "hard": True}
                ]
            }
        )
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 10,
                "order": 1,
                "integration": "rk",
            }
        )
        sol = tc.solve_ocp()
        _, x_val = tc.sol_sample(x, grid="control")
        _, u_val = tc.sol_sample(u, grid="control")
        self.assertAlmostEqual(
            x_val[-1, 0], 0.05, 6, "final constraint on vector state not respected"
        )
        self.assertAlmostEqual(
            x_val[-1, 1], 0.1, 6, "final constraint on vector state not respected"
        )
        self.assertAlmostEqual(
            u_val[-1, 0], 1.0, 6, "final constraint on vector control not respected"
        )
        self.assertAlmostEqual(
            u_val[-1, 1], -1.0, 6, "final constraint on vector control not respected"
        )

        # testing quadratic terminal costs
        del tc
        tc = tp.task_context(2)
        x = tc.create_expression("x", "state", (2, 1))
        u = tc.create_expression("u", "control", (2, 1))
        tc.set_dynamics(x, u)
        tc.add_regularization(x, 10)
        tc.add_regularization(u, 10)
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": x, "reference": [0.0, 0.0]}]}
        )
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": u, "reference": [0.0, 0.0]}]}
        )

        tc.add_task_constraint(
            {
                "final_constraints": [
                    {
                        "expression": x,
                        "reference": [0.05, 0.1],
                        "hard": False,
                        "gain": 10,
                    }
                ]
            }
        )
        tc.add_task_constraint(
            {
                "final_constraints": [
                    {
                        "expression": u,
                        "reference": [1.0, -1.0],
                        "hard": False,
                        "gain": 10,
                    }
                ]
            }
        )
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 2,
                "order": 1,
                "integration": "rk",
            }
        )
        sol = tc.solve_ocp()
        _, x_val = tc.sol_sample(x, grid="control")
        _, u_val = tc.sol_sample(u, grid="control")
        del tc

        tc = tp.task_context(2)
        x = tc.create_expression("x", "state", (2, 1))
        u = tc.create_expression("u", "control", (2, 1))
        tc.set_dynamics(x, u)
        tc.add_regularization(x, 10)
        tc.add_regularization(u, 10)
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": x, "reference": [0.0, 0.0]}]}
        )
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": u, "reference": [0.0, 0.0]}]}
        )

        ocp = tc.ocp
        tc.add_task_constraint(
            {
                "final_constraints": [
                    {
                        "hard": False,
                        "expression": x,
                        "reference": [0.05, 0.1],
                        "gain": 10,
                    }
                ]
            }
        )
        tc.add_task_constraint(
            {
                "final_constraints": [
                    {
                        "hard": False,
                        "expression": u,
                        "reference": [1.0, -1.0],
                        "gain": 10,
                    }
                ]
            }
        )
        # ocp.add_objective(ocp.at_tf(cs.sumsqr(x - [0.05, 0.1])) * 10)
        # ocp.add_objective(ocp.at_tf(cs.sumsqr(u - [1.0, -1.0])) * 10)
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 2,
                "order": 1,
                "integration": "rk",
            }
        )
        sol = tc.solve_ocp()
        _, x_val2 = tc.sol_sample(x, grid="control")
        _, u_val2 = tc.sol_sample(u, grid="control")
        self.assertAlmostEqual(
            x_val[-1, 0],
            x_val2[-1, 0],
            6,
            "terminal quadratic constraint on vector state not respected",
        )
        self.assertAlmostEqual(
            x_val[-1, 1],
            x_val2[-1, 1],
            6,
            "terminal quadratic constraint on vector state not respected",
        )
        self.assertAlmostEqual(
            u_val[-1, 0],
            u_val2[-1, 0],
            6,
            "final constraint on vector control not respected",
        )
        self.assertAlmostEqual(
            u_val[-1, 1],
            u_val[-1, 1],
            6,
            "final constraint on vector control not respected",
        )

        # testing L1 terminal costs
        del tc
        tc = tp.task_context(2)
        x = tc.create_expression("x", "state", (2, 1))
        u = tc.create_expression("u", "control", (2, 1))
        tc.set_dynamics(x, u)
        tc.add_regularization(x, 10)
        tc.add_regularization(u, 10)
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": x, "reference": [0.0, 0.0]}]}
        )
        tc.add_task_constraint(
            {"initial_constraints": [{"expression": u, "reference": [0.0, 0.0]}]}
        )

        tc.add_task_constraint(
            {
                "final_constraints": [
                    {
                        "expression": x,
                        "reference": [0.05, 0.1],
                        "norm": "L1",
                        "hard": False,
                        "gain": 1,
                    }
                ]
            }
        )
        tc.add_task_constraint(
            {
                "final_constraints": [
                    {
                        "expression": u,
                        "reference": [1.0, -1.0],
                        "hard": False,
                        "gain": 1,
                    }
                ]
            }
        )
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 2,
                "order": 1,
                "integration": "rk",
            }
        )
        sol = tc.solve_ocp()
        _, x_val = tc.sol_sample(x, grid="control")
        _, u_val = tc.sol_sample(u, grid="control")

        tc2 = tp.task_context(2)
        x2 = tc2.create_expression("x", "state", (2, 1))
        u2 = tc2.create_expression("u", "control", (2, 1))
        tc2.set_dynamics(x2, u2)
        tc2.add_regularization(x2, 10)
        tc2.add_regularization(u2, 10)
        tc2.add_task_constraint(
            {"initial_constraints": [{"expression": x2, "reference": [0.0, 0.0]}]}
        )
        tc2.add_task_constraint(
            {"initial_constraints": [{"expression": u2, "reference": [0.0, 0.0]}]}
        )

        ocp = tc2.stages[0]
        slack = ocp.variable(2, 1)
        ocp.subject_to(-slack <= (ocp.at_tf(x2) - [0.05, 0.1] <= slack))
        ocp.add_objective((slack[0] + slack[1]) * 1)
        ocp.add_objective(ocp.at_tf(cs.sumsqr(u2 - [1.0, -1.0])) * 1)
        tc2.set_ocp_solver("ipopt", solver_options)
        tc2.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 2,
                "order": 1,
                "integration": "rk",
            }
        )
        sol2 = tc2.solve_ocp()
        _, x_val2 = tc2.sol_sample(x2)
        _, u_val2 = tc2.sol_sample(u2)
        self.assertAlmostEqual(
            x_val[-1, 0],
            x_val2[-1, 0],
            6,
            "terminal quadratic constraint on vector state not respected",
        )
        self.assertAlmostEqual(
            x_val[-1, 1],
            x_val2[-1, 1],
            6,
            "terminal quadratic constraint on vector state not respected",
        )
        self.assertAlmostEqual(
            u_val[-1, 0],
            u_val2[-1, 0],
            6,
            "final constraint on vector control not respected",
        )
        self.assertAlmostEqual(
            u_val[-1, 1],
            u_val[-1, 1],
            6,
            "final constraint on vector control not respected",
        )

    def test_stage(self):
          # Testing the addition of the final constraints
        solver_options = {
            "ipopt": {"print_level": 0},
            "print_time": False,
            "expand": False,
        }
        # testing hard terminal constraints on scalars
        tc = tp.task_context(2)
        x = tc.create_expression("x", "state", (1, 1))
        u = tc.create_expression("u", "control", (1, 1))
        tc.set_dynamics(x, u)

        tc.add_task_constraint(
            {"final_constraints": [{"expression": x, "reference": 0.05, "hard": True}]}
        )
        tc.add_task_constraint(
            {"final_constraints": [{"expression": u, "reference": 1.0, "hard": True}]}
        )
        tc.set_ocp_solver("ipopt", solver_options)
        tc.set_discretization_settings(
            {
                "discretization method": "multiple shooting",
                "horizon size": 10,
                "order": 1,
                "integration": "rk",
            }
        )

        stage2 = tc.create_stage(time=None, horizon_steps=15, time_init_guess=6)
        self.assertEqual(len(tc.stages), 2)

        stage3 = tc.create_stage(time=1, horizon_steps=15, time_init_guess=6)
        self.assertEqual(len(tc.stages), 3)

        p = tc.create_expression("p", "parameter", (1, 1))
        self.assertTrue('p' in tc.parameters)

        state0 = tc.create_state(name='state0', shape=(1, 1), init_parameter=True, warm_start=1, stage=2)
        self.assertTrue('state0' in tc.states)

        parameter0 = tc.create_parameter(name='parameter0', shape=(1, 1), port_or_property=1, stage=2, grid=None)
        self.assertTrue('parameter0' in tc.parameters)

        parameter1 = tc.create_parameter(name='parameter1', shape=(1, 1), port_or_property=2, stage=2, grid='integrator')
        self.assertTrue('parameter1' in tc.parameters)

        control0 = tc.create_control(name='control0', shape=(1, 1), outport=True, stage=0)
        self.assertTrue('control0' in tc.controls)

        tc.add_objective(cs.sumsqr(parameter0), stage=2)

        tc.add_task_constraint(
            {"final_constraints": [{"expression": x, "inequality": True, 'upper_limits':1,  "hard": True}]}
        )

        tc.add_task_constraint(
            {"final_constraints": [{"expression": x, 'lub':True, 'upper_limits':1, 'lower_limits':0,  "hard": True}]}
        )

        # params = tc.get_parameters()

    def test_MPC_component(self):
        from tasho import robot as rob
        from tasho import input_resolution
        import numpy as np
        import copy

        horizon_size = 10
        max_joint_acc = 500 * 3.14159 / 180
        max_joint_vel = 40 * 3.14159 / 180
        time_optimal = True
        horizon_period = 2 #in seconds
        pi = 3.14159
        q0_start = [0.0, -pi/3, pi/3, -pi/2, -pi/2, 0.0]
        q0_start = cs.DM([-0.55953752222222, -0.92502372222222, 1.1693696111111, -1.6929679444444, -1.570795, 0])
        obstacle_clearance = 0.03

        obs_cyl = {'radius':0.15, 'z_max': 0.4, 'center':[0.95*10, 0.00], 'z_min':0.0}
        cyl_bottom = [obs_cyl['center'][0], obs_cyl['center'][1], obs_cyl['z_min']]
        cyl_top = [obs_cyl['center'][0], obs_cyl['center'][1], obs_cyl['z_max']]
        end_effector_height = 0.3
        approach_distance = 0.075
        box1 = {'b_height':0.3, 'b_max_x':1.0, 'b_max_y':-0.2, 'b_min_x':0.4, 'b_min_y':-0.8}
        box2 = {'b_height':0.3, 'b_max_x':1.0, 'b_max_y':0.8, 'b_min_x':0.4, 'b_min_y':0.2}

        box_start = box1
        box_dest = box2

        robot = rob.Robot("ur10")

        robot.set_joint_acceleration_limits(lb=-max_joint_acc, ub=max_joint_acc)
        robot.set_joint_velocity_limits(lb=-max_joint_vel, ub=max_joint_vel)


        if time_optimal:
            tc = tp.task_context(horizon_steps = horizon_size)
        else:
            tc = tp.task_context(time= horizon_period, horizon_steps = horizon_size)


        q1, q_dot1, q_ddot1, q_init1, q_dot_init1 = input_resolution.acceleration_resolved(tc, robot, {}, stage = 0)

        # Computing the FK
        fk_vals1 = robot.fk(q1)[6]
        fk_ee = fk_vals1
        # Adding the gripper height to get the tip position
        fk_ee[0:3, 3] += fk_vals1[0:3, 0:3]@cs.DM([end_effector_height, 0, 0])
        fk_ee_fun = cs.Function('fk_ee', [q1], [fk_ee])

        T_goal = np.array(
            [
                [0.0, 0.0, 1.0, 0.4],
                [0.0, 1.0, 0.0, 0.6],
                [-1.0, 0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # final_vel = {"hard": True, "expression": q_dot1, "reference": 0}
        final_pose_z = {"hard": True, "inequality":True, "expression": -fk_ee[2,3], "upper_limits": -box_dest['b_height']}
        final_pose_x = {"hard":True, "lub":True, "expression":fk_ee[0,3], "upper_limits":box_dest["b_max_x"], "lower_limits":box_dest["b_min_x"]}
        final_pose_y = {"hard":True, "lub":True, "expression":fk_ee[1,3], "upper_limits":box_dest["b_max_y"], "lower_limits":box_dest["b_min_y"]}
        tc.add_task_constraint({"final_constraints":[final_pose_x, final_pose_y, final_pose_z]}, stage = 0)

        #Adding obstacle avoidance constraints
        sep_hyp1 = tc.create_control('sep_hyp', (4,1))
        # sep_hyp1 = cs.vertcat(sep_hyp1, cs.sqrt(1 - cs.sumsqr(sep_hyp1[1:3])))
        obs_to_sep1 = sep_hyp1[0] + sep_hyp1[1:].T@cs.DM(cyl_bottom)
        obs_to_sep2 = sep_hyp1[0] + sep_hyp1[1:].T@cs.DM(cyl_top)
        ee_to_sep = sep_hyp1[0] + sep_hyp1[1:].T@fk_ee[0:3,3]
        flange_to_sep = sep_hyp1[0] + sep_hyp1[1:].T@fk_vals1[0:3,3]

        #separating hyperplane unit norm
        sh1_unit_norm = {"hard":True, "inequality":True, "expression":cs.sumsqr(sep_hyp1[1:]), "upper_limits":1.0, "include_first":True}
        sh1_con_obs = {"hard":True, "inequality":True, "expression":cs.vertcat(obs_to_sep1, obs_to_sep2), "upper_limits":-obs_cyl['radius'] - obstacle_clearance, "include_first":True}
        sh1_con_rob = {"hard":True, "inequality":True, "expression":-cs.vertcat(ee_to_sep, flange_to_sep), "upper_limits": - obstacle_clearance, "include_first":True}
        tc.add_task_constraint({"path_constraints":[sh1_unit_norm, sh1_con_obs, sh1_con_rob]})

        tc.add_regularization(q_dot1, 1e-3)
        tc.minimize_time(10, 0)
        tc.set_value(q_init1, q0_start, stage = 0)
        tc.set_value(q_dot_init1, [0]*6, stage = 0)
        tc.set_discretization_settings({})
        tc.set_ocp_solver("ipopt")

        #creating the second stage: reaching the desired approach motion start point
        if not time_optimal:
            stage2 = tc.create_stage(time = 2, horizon_steps = 5)
        else:
            stage2 = tc.create_stage(horizon_steps = 5)

        q2, q_dot2, q_ddot2 = input_resolution.acceleration_resolved(tc, robot, {'init_parameter':False}, stage = 1)

        fk_ee2 = fk_ee_fun(q2)
        T_goal_approach = cs.DM(copy.deepcopy(T_goal))
        T_goal_approach[0:3,3] += cs.DM(T_goal_approach[0:3, 0:3])@cs.DM([-approach_distance, 0, 0])
        final_pose_trans = {"hard": True, "expression": fk_ee2[0:3,3], "reference": T_goal_approach[0:3,3]}
        #strictly enforce the direction of the axis. LICQ fails at the solution. So adding as inequality constraint for robustness
        final_pose_rot = {"hard":True, "inequality":True, "expression":-fk_ee2[0:3,0].T@T_goal_approach[0:3,0], "upper_limits":-0.99}
        tc.add_task_constraint({"final_constraints":[final_pose_trans, final_pose_rot]}, stage = 1)

        tc.ocp.subject_to(tc.stages[0].at_tf(q1) == stage2.at_t0(q2))
        tc.ocp.subject_to(tc.stages[0].at_tf(q_dot1) == stage2.at_t0(q_dot2))
        tc.add_regularization(q_dot2, 1e-3, stage = 1)
        tc.minimize_time(10, 1)

        # creating the third stage: reaching the desired position
        if not time_optimal:
            stage3 = tc.create_stage(time = 0.5, horizon_steps = 5)
        else:
            stage3 = tc.create_stage(horizon_steps = 5)

        q3, q_dot3, q_ddot3 = input_resolution.acceleration_resolved(tc, robot, {'init_parameter':False}, stage = 2)
        fk_ee3 = fk_ee_fun(q3)

        # Pose constraints
        final_pose_trans = {"hard": True, "expression": fk_ee3[0:3,3], "reference": T_goal[0:3,3]}
        #strictly enforce the direction of the axis. LICQ fails at the solution. So adding as inequality constraint for robustness
        final_pose_rot = {"hard":True, "inequality":True, "expression":-fk_ee3[0:3,0].T@T_goal_approach[0:3,0], "upper_limits":-0.99}

        #terminal zero velocity constraint
        final_vel_con = {"hard":True, "expression":q_dot2, "reference":0.0}
        tc.add_task_constraint({"final_constraints":[final_pose_trans, final_pose_rot]}, stage = 2)

        tc.add_regularization(q_dot3, 1e-3, stage = 2)
        tc.ocp.subject_to(stage2.at_tf(q2) == stage3.at_t0(q3))
        tc.ocp.subject_to(stage2.at_tf(q_dot2) == stage3.at_t0(q_dot3))
        tc.minimize_time(10, 2)

        try:
            tc.set_initial(q1, q0_start)
            tc.set_initial(q2, q0_start, stage=1)
            tc.set_initial(q3, q0_start, stage=2)
            sol = tc.solve_ocp()
        except:
            print("Caught exception, trying thrice with random initializations")
            counter = 0
            while counter < 3:
                q0_randstart = np.random.rand(6,1) - 0.5
                tc.set_initial(q1, q0_randstart)
                tc.set_initial(q2, q0_randstart, stage=1)
                tc.set_initial(q3, q0_randstart, stage=2)
                try:
                    sol = tc.solve_ocp()
                    break;
                except:
                    counter += 1


        t_grid1, q1_sol = tc.sol_sample(q1, stage = 0)
        t_grid2, q2_sol = tc.sol_sample(q2, stage = 1)
        t_grid3, q3_sol = tc.sol_sample(q3, stage = 2)

        tc.add_monitor(
            {
                "name": "termination_criteria",
                "expression": cs.sqrt(cs.sumsqr(q_dot1)) - 0.001,
                "reference": 0.0,  # doesn't matter
                "greater": True,  # doesn't matter
                "initial": True,
            }
        )

        ## Performing code-generation (to potentially deploy an orocos component)
        cg_opts = {
            "ocp_cg_opts":{"jit":False, "save":False, "codegen":False}, 
            'mpc':True,
            "mpc_cg_opts":{"jit":False, "save":False, "codegen":False}}
        varsdb = tc.generate_MPC_component(".", cg_opts)
        
        self.assertEqual(varsdb['ocp_fun_name'], 'tc_ocp')
        self.assertEqual(varsdb['mpc_fun_name'], 'tc_mpc')


    def test_path_constraints(self):
        tc = tp.task_context(5, horizon_steps=5)

        x, x0 = tc.create_state("x", init_parameter=True)
        u = tc.create_control("u")
        p = tc.create_parameter("p")

        tc.set_dynamics(x, u)

        task_spec = {}
        task_spec["initial_constraints"] = [
            {"expression": u, "lub": True, "hard": True, 'lower_limits':-1, 'upper_limits': 1, "gain": 1},
            {"expression": x, "reference": 0, "hard": True, "gain": 1}
        ]
        task_spec["path_constraints"] = [
            {"expression": u, "reference": 0, "hard": False, "gain": 1, "name": "u_0"},
            {"expression": x, "reference": 0, "hard": False, "gain": 1e-3, "norm" : "L1", "name": "x_0"},
            {"expression": p, "reference": 0, "hard": False, "gain": 1e-5, "norm" : "L2_nonsquared", "name": "p_0"},
            {"expression": x + p, "reference": 0, "hard": True},
            {"expression": u, "inequality": True, "hard": True, "upper_limits":1, "lower_limits":-1},
            {"expression": u + p, "inequality": True, "hard": False, "norm" : "L2", "upper_limits":5, "name": "u_p_5", 'gain': 3},
            {"expression": u + p, "inequality": True, "hard": False, "norm" : "L1", "upper_limits":5, "name": "u_p_5_L1", 'gain': 3},
            {"expression": u + p, "inequality": True, "hard": False, "norm" : "squaredL2", "upper_limits":5, "name": "u_p_5_sqL2", 'gain': 3, 'slack_name': 's_0'},
            {"expression": u, "lub": True, "hard": False, "norm" : "L2", "upper_limits":1, "lower_limits":-1, "name": "u_lub", 'gain': 3},
            {"expression": u, "lub": True, "hard": False, "norm" : "L1", "upper_limits":1, "lower_limits":-1, "name": "u_lub_L1", 'gain': 3},
        ]
        task_spec["final_constraints"] = [
            {"expression": x ** 2, "reference": p, "hard": True},
            {"expression": x - p, "reference": 1, "hard": False, "norm": "L2", "name":'L2_x', "gain": 1},
            {"expression": p - u, "reference": 1, "hard": False, "norm": "L1", "name":'L1_x', "gain": 1}
        ]

        tc.add_task_constraint(task_spec)
        

        self.assertTrue(tc.constraints['L2_x'] is not None)
        self.assertTrue(tc.constraints['L1_x'] is not None)
        self.assertTrue(tc.constraints['x_0'] is not None)
        self.assertTrue(tc.constraints['p_0'] is not None)
        self.assertTrue(tc.constraints['u_p_5'] is not None)
        self.assertTrue(tc.constraints['u_p_5_L1'] is not None)
        self.assertTrue(tc.constraints['u_lub_L1'] is not None)


        solver_options = {
            "ipopt": {"print_level": 0},
            "print_time": False,
            "expand": True,
        }

        tc.set_ocp_solver("ipopt", solver_options)
        disc_settings = {
            "discretization method": "multiple shooting",
            "order": 2,
            "integration": "rk",
        }
        tc.set_discretization_settings(disc_settings)

        disc_settings = {
            "discretization method": "single shooting",
            "order": 2,
            "integration": "rk",
        }
        tc.set_discretization_settings(disc_settings)

        disc_settings = {
            "discretization method": "direct collocation",
            "order": 2,
            "integration": "rk",
        }
        tc.set_discretization_settings(disc_settings)
        

        # Testing the addition of path constraints. compare it with opti that uses multiple-shooting
        # print("Not implemented")

    def test_util_functions(self):
        tc = tp.task_context(5, horizon_steps=5)

        x, x0 = tc.create_state("x", init_parameter=True)
        u = tc.create_control("u")
        p = tc.create_parameter("p")

        tc.set_dynamics(x, u)

        task_spec = {}
        task_spec["path_constraints"] = [
            {"expression": u, "reference": 0, "hard": False, "gain": 1}
        ]
        task_spec["final_constraints"] = [
            {"expression": x ** 2, "reference": p, "hard": True}
        ]

        tc.add_task_constraint(task_spec)

        solver_options = {
            "ipopt": {"print_level": 0},
            "print_time": False,
            "expand": True,
        }

        tc.set_ocp_solver("ipopt", solver_options)
        disc_settings = {
            "discretization method": "multiple shooting",
            "order": 2,
            "integration": "rk",
        }
        tc.set_discretization_settings(disc_settings)

        tc.set_value(p, 5)
        tc.set_value(x0, 0.05)
        sol = tc.solve_ocp()

        primal_residual = tc.function_primal_residual()
        self.assertEqual(primal_residual.name(), 'fun_pr')

if __name__ == "__main__":
    unittest.main()
