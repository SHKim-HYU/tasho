## OCP for point-to-point motion and visualization of a KUKA robot arm

from tasho import task_prototype_rockit as tp
from tasho import input_resolution, world_simulator
from tasho import robot as rob
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from tasho.utils import dist_computation
import pybullet as p
import time, copy

if __name__ == "__main__":

    print("Task specification and visualization of P2P OCP")

    horizon_size = 20
    t_ocp = 0.15
    max_joint_acc = 120 * 3.14159 / 180
    max_joint_vel = 120 * 3.14159 / 180

    # Different OCP options
    time_optimal = False
    coll_avoid = True
    frame_constraint = False
    robot_choice = "iiwa7"  # "kinova"  #
    ocp_control = "acceleration_resolved"  # "torque_resolved"  #
    L1_pathcost = False  # Heuristically time optimal solution

    robot = rob.Robot(robot_choice)

    robot.set_joint_acceleration_limits(lb=-max_joint_acc, ub=max_joint_acc)
    robot.set_joint_velocity_limits(lb=-max_joint_vel, ub=max_joint_vel)

    if time_optimal:
        tc = tp.task_context()
        tc.minimize_time(100)
    else:
        tc = tp.task_context(t_ocp * horizon_size, horizon_size)

    if ocp_control == "acceleration_resolved":
        q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(
            tc, robot, {}
        )
    elif ocp_control == "torque_resolved":
        q, q_dot, q_ddot, tau, q0, q_dot0 = input_resolution.torque_resolved(
            tc, robot, {"forward_dynamics_constraints": False}
        )

    # computing the expression for the final frame
    if robot_choice == "iiwa7":
        fk_vals = robot.fk(q)[6]
        # T_goal = np.array([[-1., 0., 0., 0.4],
        # [0., 1., 0., -0.1],
        # [0.0, 0., -1.0, 0.4],
        # [0.0, 0.0, 0.0, 1.0]])
        T_goal = np.array(
            [
                [-1.0, 0.0, 0.0, 0.4],
                [0.0, 1.0, 0.0, -0.6],
                [0.0, 0.0, -1.0, 0.4],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # T_goal = np.array([[1., 0., 0., 0.44],
        # [0., 1., 0., 0.5],
        # [0.0, 0., 1.0, 0.5],
        #  [0.0, 0.0, 0.0, 1.0]])
    elif robot_choice == "kinova":
        # T_goal for Kinova
        fk_vals = robot.fk(q)[6]
        T_goal = np.array(
            [[0, 1, 0, 0.6], [-1, 0, 0, 0.2], [0, 0, 1, 0.5], [0, 0, 0, 1]]
        )
    # q0_val = [0]*7
    # q0_val = [
    #     0.6967786678678314,
    #     1.0571249256028108,
    #     0.14148034853277666,
    #     -1.270205899164967,
    #     0.24666659678004457,
    #     0.7847437220601475,
    #     0.41090241207031053,
    # ]
    q0_val = [
        1.0193752249977548,
        -0.05311582280659044,
        -2.815452580946695,
        1.3191046402052224,
        2.8582660722530533,
        1.3988994390898029,
        1.8226311094569714,
    ]
    # q0_val = robot.generate_random_configuration()
    # q0_val = [
    #     1.4280132556823228,
    #     1.4328754849546814,
    #     -0.8174926922820074,
    #     1.6300424942291611,
    #     1.5253416457541626,
    #     0.40308575415053083,
    #     -0.7050632112780262,
    # ]
    # q0_val = [
    #     2.4884284478753274,
    #     1.8410163304680323,
    #     -2.4614376640225717,
    #     -1.8163528893845817,
    #     2.67404192365677,
    #     -1.2615242048333508,
    #     -0.6404874946679473,
    # ]

    tc.ocp.set_value(q0, q0_val)
    tc.ocp.set_value(q_dot0, [0] * 7)
    tc.ocp.set_initial(q, q0_val)
    np.random.seed(0)
    q_init_random = robot.generate_random_configuration()
    # q_init_random = [-1.2925769493873958, 1.6948105720448297, -1.8663904352037908, 0.6752120689663585, -0.169008150759685, 0.1202290569932778, 2.147437378108014]
    # tc.ocp.set_initial(q, q_init_random)
    final_vel = {"hard": True, "expression": q_dot, "reference": 0}
    if frame_constraint:
        # final_pos = {'hard':True, 'type':'Frame', 'expression':fk_vals, 'reference':T_goal}
        final_pos = {
            "hard": False,
            "type": "Frame",
            "expression": fk_vals,
            "reference": T_goal,
            "rot_gain": 10,
            "trans_gain": 10,
            "norm": "L1",
        }
        final_constraints = {"final_constraints": [final_pos, final_vel]}
    else:
        final_position = {
            "hard": True,
            "expression": fk_vals[0:3, 3],
            "reference": T_goal[0:3, 3],
            "gain": 10,
            "norm": "L1",
        }
        final_orientation = {
            "hard": True,
            "expression": fk_vals[0:3, 2].T @ T_goal[0:3, 2],
            "reference": 1,
            "gain": 10,
            "norm": "L1",
        }
        final_constraints = {
            "final_constraints": [final_position, final_orientation, final_vel]
        }

    tc.add_task_constraint(final_constraints)

    # adding penality terms on joint velocity and position
    vel_regularization = {
        "hard": False,
        "expression": q_dot,
        "reference": 0,
        "gain": 1e-3,
    }
    if ocp_control == "acceleration_resolved":
        control_regularization = {
            "hard": False,
            "expression": q_ddot,
            "reference": 0,
            "gain": 1e-3,
        }
    elif ocp_control == "torque_resolved":
        control_regularization = {
            "hard": False,
            "expression": tau,
            "reference": 0,
            "gain": 1e-6,
        }
    if L1_pathcost:
        path_pos = {
            "hard": False,
            "type": "Frame",
            "expression": fk_vals,
            "reference": T_goal,
            "rot_gain": 0.5,
            "trans_gain": 0.5,
            "norm": "L1",
        }
        task_objective = {
            "path_constraints": [vel_regularization, control_regularization, path_pos]
        }
    else:
        task_objective = {
            "path_constraints": [vel_regularization, control_regularization]
        }
    tc.add_task_constraint(task_objective)

    # Add collision avoidance constraints, if activated
    if coll_avoid:
        kuka_envelope_fun = cs.Function.load(
            "./models/robots/KUKA/iiwa7/kuka_ball_envelope.casadi"
        )
        envelopes = kuka_envelope_fun(q)
        ball_obs = {"center": [0.3, 0.0, 1.1], "radius": 0.1}
        cube2 = {
            "tf": np.array(
                [[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 1.1], [0, 0, 0, 1]]
            ),
            "xyz_len": [0.1, 0.1, 0.1],
        }
        cube = {}
        cube["tf"] = np.array(
            [[1, 0, 0, 0.6], [0, 1, 0, 0], [0, 0, 1, 0.4], [0, 0, 0, 1]]
        )
        cube["xyz_len"] = np.array([0.1, 0.1, 0.4])
        cube3 = copy.deepcopy(cube)
        cube3["tf"][0, 3] += 2.0

        cube4 = copy.deepcopy(cube)
        cube4["tf"][0, 3] -= 2.0

        cube5 = copy.deepcopy(cube)
        cube5["tf"][1, 3] -= 2.0

        cube6 = copy.deepcopy(cube)
        cube6["tf"][2, 3] += 10.0

        softmax = False
        no_obs = 50
        distances = cs.MX.zeros(5 * no_obs, 1)
        for i in range(1, 6):
            ball = {"center": envelopes[0:3, i], "radius": envelopes[3, i]}
            distance = dist_computation.dist_sphere_box(ball, cube, do_softmax=softmax)
            distances[(i - 1) * no_obs] = distance
            distance2 = dist_computation.dist_sphere_box(
                ball, cube2, do_softmax=softmax
            )
            distances[(i - 1) * no_obs + 1] = distance2
            distance3 = dist_computation.dist_sphere_box(
                ball, cube3, do_softmax=softmax
            )
            distances[(i - 1) * no_obs + 2] = distance3
            distance4 = dist_computation.dist_sphere_box(
                ball, cube4, do_softmax=softmax
            )
            distances[(i - 1) * no_obs + 3] = distance4
            distance5 = dist_computation.dist_sphere_box(
                ball, cube5, do_softmax=softmax
            )
            distances[(i - 1) * no_obs + 4] = distance5

            for j in range(no_obs - 5):
                # cube6["tf"][0, 3] += np.random.randn()
                # cube6["tf"][1, 3] += np.random.randn()
                # cube6["tf"][2, 3] += np.random.randn()
                ball_obs["center"] = list(cube6["tf"][0:3, 3])
                # distance6 = dist_computation.dist_sphere_box(
                #     ball, cube6, do_softmax=softmax
                # )
                distance6 = dist_computation.dist_spheres(ball, ball_obs)
                distances[(i - 1) * no_obs + 5 + j] = distance6

        # min_distance = -dist_computation.softmax(-distances, alpha=200)
        min_distance = distances
        tc.add_task_constraint(
            {
                "path_constraints": [
                    {
                        "inequality": True,
                        "hard": True,
                        "expression": -min_distance,
                        "upper_limits": -0.04,
                        "gain": 100,
                        "norm": "L1",
                    }
                ]
            }
        )

    # tc.set_ocp_solver("ipopt", {"ipopt": {"linear_solver": "ma27", "tol": 1e-3}})
    tc.set_ocp_solver(
        "ipopt",
        {
            "ipopt": {
                "max_iter": 1000,
                "linear_solver": "ma27",
                "hessian_approximation": "limited-memory",
                "limited_memory_max_history": 5,
                "tol": 1e-3,
            }
        },
    )
    # disc_settings = {
    #     "discretization method": "multiple shooting",
    #     "order": 1,
    #     "integration": "rk",
    # }
    tc.set_discretization_settings({})
    sol = tc.solve_ocp()

    ts, q_sol = sol.sample(q, grid="control")
    ts, q_dot_sol = sol.sample(q_dot, grid="control")

    # check feasibility if soft constraints were used
    print("Initial joint position")
    print(q0_val)
    if True:
        rot_err = robot.fk(q_sol[-1, :])[6][0:3, 0:3].T @ T_goal[0:3, 0:3]
        assert cs.norm_1(robot.fk(q_sol[-1, :])[6][0:3, 3] - T_goal[0:3, 3]) <= 1e-3
        if frame_constraint:
            assert (
                cs.fabs(rot_err[0, 0] - 1)
                + cs.fabs(rot_err[1, 1] - 1)
                + cs.fabs(rot_err[2, 2] - 1)
                <= 1e-4
            )
        else:
            assert cs.fabs(rot_err[2, 2] - 1) <= 1e-4
        if coll_avoid:
            _, dist1_sol = sol.sample(distance, grid="control")
            print("distance_solution = " + str(dist1_sol))
            assert (dist1_sol >= 0.01).all()
            _, dist2_sol = sol.sample(distance2, grid="control")
            # assert (dist2_sol >= 0.01).all()
            print(robot.fk(q_sol[-1, :])[6][0:3, 0:3] @ T_goal[0:3, 0:3])

    # visualizing the results
    obj = world_simulator.world_simulator(plane_spawn=True, bullet_gui=True)
    obj.visualization_realtime = True
    position = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0, 1.0]
    kukaID = obj.add_robot(position, orientation, robot_choice)

    if coll_avoid:
        cube_dimensions = list(cube["xyz_len"])
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=cube_dimensions
        )
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_BOX, halfExtents=cube_dimensions
        )
        boxID = p.createMultiBody(
            200, collisionShapeId, visualShapeId, list(cube["tf"][0:3, 3]), [0, 0, 0, 1]
        )

        obj.add_cube(
            {"position": list(cube2["tf"][0:3, 3]), "orientation": [0, 0, 0, 1.0]},
            scale=0.1 * 1.8,
            fixedBase=True,
        )

    if time_optimal:
        t_ocp = sol.value(tc.ocp.T) / horizon_size

    joint_indices = [0, 1, 2, 3, 4, 5, 6]
    obj.resetJointState(kukaID, joint_indices, q0_val)

    s_qsol = sol.sampler(q_dot)
    t_sol = np.arange(0, sol.value(tc.ocp.T), obj.physics_ts)
    q_dot_sol = s_qsol(t_sol)

    for i in range(len(t_sol) - 1):
        q_vel_current = 0.5 * (q_dot_sol[i] + q_dot_sol[i + 1])
        obj.setController(
            kukaID, "velocity", joint_indices, targetVelocities=q_vel_current
        )
        obj.run_simulation(1)

    time.sleep(1)
    obj.end_simulation()
    # print(q_sol)
    # print the final end effector position
    print(robot.fk(q_sol[-1, :])[6])
    # print(q0_val)
    # print(q_init_random)
    print(t_ocp)
