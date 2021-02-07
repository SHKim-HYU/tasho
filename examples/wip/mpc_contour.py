### OCP for point-to-point motion and visualization of a KUKA robot arm

from tasho import task_prototype_rockit as tp
from tasho import input_resolution, world_simulator
from tasho import robot as rob
from tasho import MPC
from tasho.utils import geometry
from casadi import pi, cos, sin, acos
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

print("Task specification and visualization of contour-following example with MPC")

##########################################
# Define robot and initial joint angles
##########################################
# Import the robot object from the robot's repository (includes functions for FD, ID, FK, joint limits, etc)
robot_choice = "kinova"
ocp_control = "torque_resolved"  #'acceleration_resolved' #'torque_resolved'

robot = rob.Robot(robot_choice)

# Update robot's parameters if needed
if ocp_control == "acceleration_resolved":
    max_joint_acc = 240 * pi / 180
    robot.set_joint_acceleration_limits(lb=-max_joint_acc, ub=max_joint_acc)

# Define initial conditions of the robot
q_init = [0, pi / 6, 0, 4 * pi / 6, 0, -2 * pi / 6, -pi / 2]
q_dot_init = [0] * robot.ndof

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
# Task spacification - Contour following
##########################################

# Select prediction horizon and sample time for the MPC execution
horizon_size = 16
t_mpc = 0.02

# Initialize the task context object
tc = tp.task_context(horizon_size * t_mpc)

# Define the input type of the robot (torque or acceleration)
if ocp_control == "acceleration_resolved":
    q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})
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

# Set s(0) and s_dot(0) as parameters
s0 = tc.create_expression("s0", "parameter", (1, 1))
s_dot0 = tc.create_expression("s_dot0", "parameter", (1, 1))

s_init_con = {"expression": s, "reference": s0}
s_dot_init_con = {"expression": s_dot, "reference": s_dot0}
init_constraints = {"initial_constraints": [s_init_con, s_dot_init_con]}
tc.add_task_constraint(init_constraints)

# Add constraints for path-progress variable (0 <= s <= 1, s_dot >= 0)
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

# Define contour/path based on the path-progress variable s
pos_path, rot_path, sdot_path = contour_path(s)

# Define end-effector position and orientation error
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


# Set tunnel constraints to allow a deviation from the path
pos_tunnel_con = {  # pos_tunnel_con = cs.sumsqr(pos_err(q, s)) - rho^2 <= slack
    "hard": False,
    "inequality": True,
    "expression": pos_err(q, s),
    "upper_limits": 0.01 ** 2,
    "gain": 100,
    "norm": "squaredL2",
}
tunnel_constraints = {"path_constraints": [pos_tunnel_con]}
tc.add_task_constraint(tunnel_constraints)

# Define objective
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

# Add regularization terms to the objective
tc.add_regularization(expression=(s_dot - sdot_path), weight=20, norm="L2")
tc.add_regularization(expression=pos_err(q, s), weight=1e-1, norm="L2")
tc.add_regularization(expression=rot_err(q, s), weight=1e-1, norm="L2")

if ocp_control == "torque_resolved":
    tc.add_regularization(
        expression=tau, weight=4e-5, norm="L2", variable_type="control", reference=0
    )
if ocp_control == "acceleration_resolved":
    tc.add_regularization(
        expression=q_ddot, weight=1e-3, norm="L2", variable_type="control", reference=0,
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

################################################
# Set solver and discretization options
################################################
tc.set_ocp_solver("ipopt")

disc_settings = {
    "discretization method": "multiple shooting",
    "horizon size": horizon_size,
    "order": 1,
    "integration": "rk",
}
tc.set_discretization_settings(disc_settings)

################################################
# Set parameter values
################################################
tc.ocp.set_value(q0, q_init)
tc.ocp.set_value(q_dot0, [0] * 7)
tc.ocp.set_value(s0, 0)
tc.ocp.set_value(s_dot0, 0)

################################################
# Solve the OCP that describes the task
################################################
sol = tc.solve_ocp()

################################################
# MPC Simulation
################################################
use_MPC_class = True

if use_MPC_class:

    # Create world simulator based on pybullet
    from tasho import world_simulator
    import pybullet as p

    obj = world_simulator.world_simulator(bullet_gui=True)

    # Add robot to the world environment
    position = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0, 1.0]
    kinovaID = obj.add_robot(position, orientation, "kinova")

    # Determine number of samples that the simulation should be executed
    no_samples = int(t_mpc / obj.physics_ts)
    if no_samples != t_mpc / obj.physics_ts:
        print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

    # Correspondence between joint numbers in bullet and OCP
    joint_indices = [0, 1, 2, 3, 4, 5, 6]

    # Begin the visualization by applying the initial control signal
    obj.resetJointState(kinovaID, joint_indices, q_init)
    obj.setController(kinovaID, "velocity", joint_indices, targetVelocities=q_dot_init)

    # Define MPC parameters
    mpc_params = {"world": obj}

    q0_params_info = {
        "type": "joint_position",
        "joint_indices": joint_indices,
        "robotID": kinovaID,
    }
    q_dot0_params_info = {
        "type": "joint_velocity",
        "joint_indices": joint_indices,
        "robotID": kinovaID,
    }
    s0_params_info = {"type": "progress_variable", "state": True}
    s_dot0_params_info = {"type": "progress_variable", "state": True}

    mpc_params["params"] = {
        "q0": q0_params_info,
        "q_dot0": q_dot0_params_info,
        "s0": s0_params_info,
        "s_dot0": s_dot0_params_info,
        "robots": {kinovaID: robot},
    }
    mpc_params["disc_settings"] = disc_settings
    mpc_params["solver_name"] = "sqpmethod"
    mpc_params["solver_params"] = {"qrqp": True}
    mpc_params["t_mpc"] = t_mpc
    mpc_params["control_type"] = "joint_velocity"  #'joint_torque'
    mpc_params["control_info"] = {
        "robotID": kinovaID,
        "discretization": "constant_acceleration",
        "joint_indices": joint_indices,
        "no_samples": no_samples,
    }

    # Create monitor to check some termination criteria
    tc.add_monitor(
        {
            "name": "termination_criteria",
            "expression": s,
            "reference": 0.99,
            "greater": True,
            "initial": True,
        }
    )

    # Initialize MPC object
    sim_type = "bullet_notrealtime"

    mpc_obj = MPC.MPC(tc, sim_type, mpc_params)
    mpc_obj.max_mpc_iter = 400

    # Run the ocp with IPOPT once to get a good initial guess for the MPC
    mpc_obj.configMPC_fromcurrent()

    # Execute the MPC loop
    mpc_obj.runMPC()
