import sys

from tasho import task_prototype_rockit as tp
from tasho import input_resolution, world_simulator
from tasho import robot as rob
from tasho import MPC
from tasho.utils import geometry
from tasho.utils.lin import lin

import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np
import matplotlib.pyplot as plt


robot_choice = "yumi"
ocp_control = "acceleration_resolved"  #'torque_resolved'

robot = rob.Robot(robot_choice, analytical_derivatives=True)

if ocp_control == "acceleration_resolved":
    max_joint_acc = 240 * cs.pi / 180
    robot.set_joint_acceleration_limits(lb=-max_joint_acc, ub=max_joint_acc)


if robot_choice == "yumi":
    left_arm_q_init = [-1.35, -8.72e-01, 2.18, 6.78e-01, 2.08, -9.76e-01, -1.71, 1.65e-03, 1.65e-03]
    # Right arm in home configuration
    right_arm_q_init = [0, -2.26, -2.35, 0.52, 0.025, 0.749, 0, 0, 0]

    # q_init = np.array(left_arm_q_init + right_arm_q_init).T
    q_init = np.array([-1.35488912e+00, -8.72846052e-01, 2.18411843e+00,  6.78786296e-01,
            2.08696971e+00, -9.76390128e-01, -1.71721329e+00,  1.65969745e-03,
            1.65969745e-03,  1.47829337e+00, -5.24943547e-01, -1.95134781e+00,
            5.30517837e-01, -2.69960026e+00, -8.14070355e-01,  1.17172289e+00,
            2.06459136e-03,  2.06462524e-03]).T
elif robot_choice == "kinova":
    q_init = [0, -0.523598, 0, 2.51799, 0, -0.523598, -1.5708]
    
q_dot_init = [0] * robot.ndof

horizon_size = 16
t_mpc = 0.01

tc = tp.task_context(horizon_size * t_mpc, horizon_steps=horizon_size)

if ocp_control == "acceleration_resolved":
    q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})
elif ocp_control == "torque_resolved":
    q, q_dot, q_ddot, tau, q0, q_dot0 = input_resolution.torque_resolved(
        tc, robot, {"forward_dynamics_constraints": False}
    )

s = tc.create_expression("s", "state", (1, 1))
s_dot = tc.create_expression("s_dot", "state", (1, 1))
s_ddot = tc.create_expression("s_ddot", "control", (1, 1))

tc.set_dynamics(s, s_dot)
tc.set_dynamics(s_dot, s_ddot)

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

def contour_path(s):
    ee_fk_init = robot.fk(q_init)[7]
    ee_pos_init = ee_fk_init[:3, 3]
    ee_rot_init = ee_fk_init[:3, :3]

    sdotref = 0.1
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
        0,
        a_p * cs.sin(s * (2 * cs.pi)),
        a_p * cs.sin(s * (2 * cs.pi)) * cs.cos(s * (2 * cs.pi)),
    )
    rot_path = ee_rot_init
    # rot_path = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    return pos_path, rot_path, sdot_path

# Define contour/path based on the path-progress variable s
pos_path, rot_path, sdot_path = contour_path(s)

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

tun_tunnel_con = { 
    "hard": False,
    "inequality": True,
    "expression": cs.vertcat(pos_err(q, s), rot_err(q, s)),
    "upper_limits": 0.01 ** 2,
    "gain": 100,
    "norm": "squaredL2",
}
tunnel_constraints = {"path_constraints": [tun_tunnel_con]}
tc.add_task_constraint(tunnel_constraints)

# Define objective

tc.add_regularization(expression=s_dot, reference=sdot_path, weight=20, norm="L2")
tc.add_regularization(expression=pos_err(q, s), weight=1e-1, norm="L2")
tc.add_regularization(expression=rot_err(q, s), weight=1e-1, norm="L2")

# Add regularization term for x
tc.add_regularization(
    expression=q[0:7], weight=1e-2, norm="L2", variable_type="state", reference=0
)
tc.add_regularization(
    expression=q_dot[0:7], weight=1e-2, norm="L2", variable_type="state", reference=0
)
tc.add_regularization(
    expression=s, weight=1e-2, norm="L2", variable_type="state", reference=0
)
tc.add_regularization(
    expression=s_dot, weight=1e-2, norm="L2", variable_type="state", reference=0
)

# Add regularization term for u
if ocp_control == "torque_resolved":
    tc.add_regularization(
        expression=tau, weight=4e-5, norm="L2", variable_type="control", reference=0
    )
if ocp_control == "acceleration_resolved":
    tc.add_regularization(
        expression=q_ddot,
        weight=1e-3,
        norm="L2",
        variable_type="control",
        reference=0,
    )
tc.add_regularization(
    expression=s_ddot, weight=4e-5, norm="L2", variable_type="control", reference=0
)


tc.add_objective(
    tc.ocp.at_tf(
        1e-5
        * cs.sumsqr(
            cs.vertcat(
                1e-2 * q[0:7],
                10 * q_dot[0:7],
                1e-2 * (1 - s),
                10 * s_dot,
                10 * pos_err(q, s),
                10 * rot_err(q, s),
            )
        )
    )
)

tc.set_ocp_solver("ipopt")

disc_settings = {
    "discretization method": "multiple shooting",
    "horizon size": horizon_size,
    "order": 1,
    "integration": "rk",
}
tc.set_discretization_settings(disc_settings)

print(q0, "\n")
tc.ocp.set_value(q0, q_init)
tc.ocp.set_value(q_dot0, [0] * robot.ndof)
tc.ocp.set_value(s0, 0)
tc.ocp.set_value(s_dot0, 0)

sol = tc.solve_ocp()

print("**** EXIT ****")
exit()


# MPC Simulation

from tasho import world_simulator
import pybullet as p

obj = world_simulator.world_simulator(bullet_gui=True)

position = [0.0, 0.0, 0.0]
orientation = [0.0, 0.0, 0.0, 1.0]
robotID = obj.add_robot(position, orientation, robot_choice)

no_samples = int(t_mpc / obj.physics_ts)

# Correspondence between joint numbers in bullet and OCP
if robot_choice == "yumi":
    joint_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9]
elif robot_choice == "kinova":
    joint_indices = [0, 1, 2, 3, 4, 5, 6]

obj.resetJointState(robotID, joint_indices, q_init)
obj.setController(robotID, "velocity", joint_indices, targetVelocities=q_dot_init)

mpc_params = {"world": obj}

q0_params_info = {
    "type": "joint_position",
    "joint_indices": joint_indices,
    "robotID": robotID,
}
q_dot0_params_info = {
    "type": "joint_velocity",
    "joint_indices": joint_indices,
    "robotID": robotID,
}
s0_params_info = {"type": "progress_variable", "state": True}
s_dot0_params_info = {"type": "progress_variable", "state": True}

mpc_params["params"] = {
    "q0": q0_params_info,
    "q_dot0": q_dot0_params_info,
    "s0": s0_params_info,
    "s_dot0": s_dot0_params_info,
    "robots": {robotID: robot},
}

mpc_params["disc_settings"] = disc_settings
mpc_params["solver_name"] = "sqpmethod"
mpc_params["solver_params"] = {"qrqp": True}
mpc_params["t_mpc"] = t_mpc
mpc_params["control_type"] = "joint_velocity"  #'joint_torque'
mpc_params["control_info"] = {
    "robotID": robotID,
    "discretization": "constant_acceleration",
    "joint_indices": joint_indices,
    "no_samples": no_samples,
}

mpc_params["log_solution"] = True

mpc_params["codegen"] = {
    "codegen": False,
    "filename": "mpc_c",
    "compilation": False,
    "compiler": "gcc",
    "flags": "-O3 -ffast-math -flto -funroll-loops -march=native -mfpmath=both -mvzeroupper",
    "use_external": False,
    "jit": False,
}

tc.add_monitor(
    {
        "name": "termination_criteria",
        "expression": s,
        "reference": 0.99,
        "greater": True,
        "initial": True,
    }
)

sim_type = "bullet_notrealtime"
mpc_obj = MPC.MPC(tc, sim_type, mpc_params)
mpc_obj.max_mpc_iter = 4000

mpc_obj.configMPC_fromcurrent()


mpc_obj.runMPC()