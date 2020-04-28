import sys
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from scipy.spatial.transform import Rotation
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp

from tasho import task_prototype_rockit
from tasho import input_resolution
from tasho.robot import Robot

# create a tuple to hold all variables from the ocp
OcpVars = namedtuple("OcpVars", "q q_dot q_ddot q0 q_dot0")

# ======================================================================
# Function to handle pick pose input file
# ======================================================================


def strs_to_floats(strs):
    return [float(s) for s in strs]


def parse_orientation(quaternion):
    return Rotation.from_quat(quaternion).as_matrix()


def parse_pick_poses(filename):
    poses = {}

    with open(filename) as file:
        # fixed rotation to end-effector frame
        # r_fix = Rotation.from_rotvec([0, np.pi, 0]).as_matrix()
        # kuka iiwa, x axis points into end-effector
        r_fix = Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()

        lines = file.readlines()
        for line in lines:
            data = line.rstrip().split(" ")
            name = data[1]
            pose = np.eye(4)
            pos_and_rot = strs_to_floats(data[2:])

            # get rotation matrix
            pose[:3, :3] = parse_orientation(pos_and_rot[3:])
            pose[:3, :3] = r_fix @ parse_orientation(pos_and_rot[3:])

            # get position vector, move the frawe a fixed distance
            # away from the object, there will be a long tool in between
            # the flange and the cube
            # kuka iiwa, x axis points into end-effector
            offset = np.array([0.2, 0, 0])
            pose[:3, 3] = pos_and_rot[:3] + pose[:3, :3] @ offset
            # pose[:3, 3] = pos_and_rot[:3]
            poses[name] = {"T": pose, "quat": pos_and_rot[3:], "pos": pos_and_rot[:3]}

    return poses


# ======================================================================
# Function to setup the ocp
# ======================================================================


def create_pose_constraint(pose_ref, fk_fun, q_dot_ref, q_dot_state):
    position_con = {
        "hard": True,
        "type": "Frame",
        "expression": fk_fun,
        "reference": pose_ref,
    }
    velocity_con = {"hard": True, "expression": q_dot_state, "reference": q_dot_ref}
    return {"final_constraints": [position_con, velocity_con]}


def create_path_constraint(q_dot_state, q_dot_ref, q_ddot_state, q_ddot_ref):
    """ Create penality terms on joint velocity and acceleration """
    vel_regularization = {
        "hard": False,
        "expression": q_dot_state,
        "reference": q_dot_ref,
        "gain": 1,
    }
    acc_regularization = {
        "hard": False,
        "expression": q_ddot_state,
        "reference": q_ddot_ref,
        "gain": 1,
    }

    return {"path_constraints": [vel_regularization, acc_regularization]}


def create_approach_task(robot, time_step, horizon_size, q_init, q_dot_init, goal_pose):
    # General settings at the top for easy access
    SOLVER = "ipopt"
    SOLVER_SETTINGS = {
        "ipopt": {
            "max_iter": 1000,
            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 5,
            "tol": 1e-3,
        }
    }
    DISC_SETTINGS = {
        "discretization method": "multiple shooting",
        "horizon size": horizon_size,
        "order": 1,
        "integration": "rk",
    }

    # setup the ocp problem
    tc = task_prototype_rockit.task_context(horizon_size * time_step)
    q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})

    # computing the expression for the final frame
    fk_vals = robot.fk(q)[7]

    # add constraint of goal pose and speed and acc penalties
    tc.add_task_constraint(create_pose_constraint(goal_pose, fk_vals, 0, q_dot))
    tc.add_task_constraint(create_path_constraint(q_dot, 0, q_ddot, 0))

    # initial values and settings
    tc.ocp.set_value(q0, q_init)
    tc.ocp.set_value(q_dot0, q_dot_init)
    tc.set_ocp_solver(SOLVER, SOLVER_SETTINGS)
    tc.set_discretization_settings(DISC_SETTINGS)

    return tc, q, q_dot, q_ddot, q0, q_dot0


# ======================================================================
# Functions to add and simulate stuff in bullet
# ======================================================================
def add_cubes_to_scene(poses: dict):
    cube_ids = {}
    for name, info in poses.items():
        # print( "info", info)
        cube_ids[name] = p.loadURDF(
            "models/objects/cube_small.urdf", info["pos"], info["quat"]
        )
    return cube_ids


def plot_reference_frame(p_handle, pose, length=0.2):
    """ Plot xyz axis in bullet """
    I = np.eye(3)
    v = pose[:3, 3]
    R = pose[:3, :3]
    p_handle.addUserDebugLine(v, v + length * R @ I[:, 0], [1, 0, 0])
    p_handle.addUserDebugLine(v, v + length * R @ I[:, 1], [0, 1, 0])
    p_handle.addUserDebugLine(v, v + length * R @ I[:, 2], [0, 0, 1])


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    VISUALIZATION_BULLET = True
    T_MPC = 0.5
    MAX_JOINT_ACC = 30 * 3.14159 / 180 * 2

    print("Random bin picking with Kinova Gen3")

    pick_poses = parse_pick_poses("examples/wip/pick_poses.irl")
    # select the first pick pose in the list to pick
    # this would be a for loop to pick them all in the future
    # pick_poses is a dictionary with the pose names as keys
    print(pick_poses.keys())
    pick_pose_name = "P1"
    print("We will be picking a cube at pose: " + pick_pose_name)

    # robot = Robot("kinova")
    robot = Robot("iiwa7")
    robot.set_joint_acceleration_limits(lb=-MAX_JOINT_ACC, ub=MAX_JOINT_ACC)

    # --------------------------------------------------------------------------
    # Approach the object up
    # --------------------------------------------------------------------------
    HORIZON_SIZE = 10
    q_init = [0] * 7
    q_dot_init = [0] * 7
    # q0_val = [2.58754704, -0.88858335, -1.51273745,  1.07067938, -0.96941473,  1.9196731, -0.9243157]

    # T_goal = np.array(
    #     [
    #         [0.0, 1.0, 0.0, 0.0],
    #         [1.0, 0.0, 0.0, -0.5],
    #         [0.0, 0.0, -1.0, 0.4],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )

    # T_goal = np.array(
    #     [
    #         [0.0, 0.0, -1.0, 0.0],
    #         [0.0, 1.0, 0.0, -0.5],
    #         [1.0, 0.0, 0.0, 0.4],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    T_goal = pick_poses[pick_pose_name]["T"]

    tc, q, q_dot, q_ddot, q0, q_dot0 = create_approach_task(
        robot, T_MPC, HORIZON_SIZE, q_init, q_dot_init, T_goal
    )
    sol = tc.solve_ocp()

    ts, q_sol = sol.sample(q, grid="control")

    # save the final joint values for the next segment of the plan
    final_qsol_approx = q_sol[-1, :]

    # --------------------------------------------------------------------------
    # Pick the object up
    # --------------------------------------------------------------------------
    HORIZON_SIZE_PICKUP = 16
    q_init_pickup = final_qsol_approx

    # T_drop = np.array(
    #     [[0, 1, 0, 0.5], [1, 0, 0, 0], [0, 0, -1, 0.4], [0, 0, 0, 1]]
    # )
    T_drop = np.array(
        [
            [0.0, 0.0, -1.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    (
        tc_pickup,
        q_pickup,
        q_dot_pickup,
        q_ddot_pickup,
        q0_pickup,
        q_dot0_pickup,
    ) = create_approach_task(
        robot, T_MPC, HORIZON_SIZE_PICKUP, q_init_pickup, q_dot_init, T_drop
    )

    sol_pickup = tc_pickup.solve_ocp()

    ts_pickup, q_sol_pickup = sol_pickup.sample(q_pickup, grid="control")

    # --------------------------------------------------------------------------
    # Demo simulation
    # --------------------------------------------------------------------------

    if VISUALIZATION_BULLET:

        from tasho import world_simulator
        import pybullet as p

        obj = world_simulator.world_simulator()

        position = [0.0, 0.0, 0.0]
        orientation = [0.0, 0.0, 0.0, 1.0]

        # robotID = obj.add_robot(position, orientation, "kinova")
        robotID = obj.add_robot(position, orientation, "iiwa7")

        picked_cube_id = p.loadURDF(
            "models/objects/cube_small.urdf",
            [0, -0.5, 0.4],
            [0.0, 0.0, 0.0, 1.0],
            globalScaling=1.0,
        )

        plot_reference_frame(p, T_goal)
        plot_reference_frame(p, T_drop)

        cube_ids = add_cubes_to_scene(pick_poses)
        picked_cube_id = cube_ids[pick_pose_name]
        tableID = p.loadURDF(
            "cube_small.urdf", [0.5, 0, 0], [0.0, 0.0, 0.0, 1.0], globalScaling=6.9
        )
        # table2ID = p.loadURDF(
        #     "cube_small.urdf", [0, -0.5, 0], [0.0, 0.0, 0.0, 1.0], globalScaling=6.9
        # )
        binID = p.loadURDF("models/objects/bin.urdf")
        # print(obj.getJointInfoArray(robotID))
        no_samples = int(T_MPC / obj.physics_ts)

        if no_samples != T_MPC / obj.physics_ts:
            print(
                "[ERROR] MPC sampling time not integer multiple of physics sampling time"
            )

        # correspondence between joint numbers in bullet and OCP determined after reading joint info of YUMI
        # from the world simulator
        joint_indices = [0, 1, 2, 3, 4, 5, 6]

        # begin the visualization of applying OCP solution in open loop
        ts, q_dot_sol = sol.sample(q_dot, grid="control")
        obj.resetJointState(robotID, joint_indices, q_init)
        obj.setController(
            robotID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
        )
        obj.run_simulation(250)  # Here, the robot is just waiting to start the task

        for i in range(HORIZON_SIZE):
            q_vel_current = 0.5 * (q_dot_sol[i] + q_dot_sol[i + 1])
            obj.setController(
                robotID, "velocity", joint_indices, targetVelocities=q_vel_current
            )
            obj.run_simulation(no_samples)
            # obj.run_continouous_simulation()

        # Simulate pick-up object ----------------------------------------------
        ts_pickup, q_dot_sol_pickup = sol_pickup.sample(q_dot_pickup, grid="control")
        obj.resetJointState(robotID, joint_indices, q_init_pickup)
        obj.setController(
            robotID, "velocity", joint_indices, targetVelocities=q_dot_sol_pickup[0]
        )
        obj.run_simulation(100)  # Here, the robot is just waiting to start the task

        p.createConstraint(
            robotID,
            6,
            picked_cube_id,
            -1,
            p.JOINT_FIXED,
            [0.0, 0.0, 1.0],
            [0.0, 0, 0.1],
            [0.0, 0.0, 0.1],
        )

        for i in range(HORIZON_SIZE):
            q_vel_current_pickup = 0.5 * (q_dot_sol_pickup[i] + q_dot_sol_pickup[i + 1])
            # q_vel_current_pickup = q_dot_sol_pickup[i]
            obj.setController(
                robotID,
                "velocity",
                joint_indices,
                targetVelocities=q_vel_current_pickup,
            )
            obj.run_simulation(
                no_samples + 62
            )  # Still have to fix this (if I leave no_samples, it won't finish the task)
            # obj.run_continouous_simulation()

        obj.setController(
            robotID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
        )
        obj.run_simulation(1000)
        # ----------------------------------------------------------------------

        # #create a constraint to attach the body to the robot (TODO: make this systematic and add to the world_simulator class)
        # #print(p.getNumBodies())
        # #get link state of EE
        # ee_state = p.getLinkState(robotID, 6, computeForwardKinematics = True)
        # print(ee_state)
        # #get link state of the cylinder
        # #cyl_state = p.getLinkState(cylID, -1, computeForwardKinematics = True)
        # #print(cyl_state)
        # p.createConstraint(robotID, 6, cylID, -1, p.JOINT_FIXED, [0., 0., 1.], [0., 0, 0.1], [0., 0., 0.1])
        # # obj.run_simulation(1000)
        #
        # obj.run_continouous_simulation()

        obj.end_simulation()
