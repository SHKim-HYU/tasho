# helper functions using Tasho to set up the variables and parameters
# for many standard cases such as velocity-resolved, acceleration-resolved and
# Torque-resolved MPCs to simplify code

# import sys
from tasho import task_prototype_rockit as tp
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
import numpy as np


def acceleration_resolved(tc, robot, options={}):

    """Function returns the expressions for acceleration-resolved control
    with appropriate position, velocity and acceleration constraints added
    to the task context.

    :param tc: The task context

    :param robot: robot The object of the robot in question

    :param options: Dictionary to pass further miscellaneous options
    """

    q = tc.create_expression(
        "q", "state", (robot.nq, 1)
    )  # joint positions over the trajectory
    q_dot = tc.create_expression("q_dot", "state", (robot.ndof, 1))  # joint velocities
    q_ddot = tc.create_expression("q_ddot", "control", (robot.ndof, 1))

    # expressions for initial joint position and joint velocity
    q0 = tc.create_expression("q0", "parameter", (robot.ndof, 1))
    q_dot0 = tc.create_expression("q_dot0", "parameter", (robot.ndof, 1))

    tc.set_dynamics(q, q_dot)
    tc.set_dynamics(q_dot, q_ddot)

    # add joint position, velocity and acceleration limits
    pos_limits = {
        "lub": True,
        "hard": True,
        "expression": q,
        "upper_limits": robot.joint_ub,
        "lower_limits": robot.joint_lb,
    }
    vel_limits = {
        "lub": True,
        "hard": True,
        "expression": q_dot,
        "upper_limits": robot.joint_vel_ub,
        "lower_limits": robot.joint_vel_lb,
    }
    acc_limits = {
        "lub": True,
        "hard": True,
        "expression": q_ddot,
        "upper_limits": robot.joint_acc_ub,
        "lower_limits": robot.joint_acc_lb,
    }
    joint_constraints = {"path_constraints": [pos_limits, vel_limits, acc_limits]}
    tc.add_task_constraint(joint_constraints)

    # adding the initial constraints on joint position and velocity
    joint_init_con = {"expression": q, "reference": q0}
    joint_vel_init_con = {"expression": q_dot, "reference": q_dot0}
    init_constraints = {"initial_constraints": [joint_init_con, joint_vel_init_con]}
    tc.add_task_constraint(init_constraints)

    return q, q_dot, q_ddot, q0, q_dot0


def velocity_resolved(tc, robot, options):

    print("ERROR: Not implemented and probably not recommended")


def torque_resolved(tc, robot, options={"forward_dynamics_constraints": False}):

    """Function returns the expressions for torque-resolved control
    with appropriate position, velocity and torque constraints added
    to the task context.

    :param tc: The task context

    :param robot: robot The object of the robot in question

    :param options: Dictionary to pass further options. Key 'forward_dynamics_constraints' is by default
    set to False. Then, joint accelerations is a constraint variable. Torque values are computed using
    inverse dynamics (usually faster) and are subject to box-constraints. When 'forward_dynamics_constraints'
    is set to True. Joint torques are control variables and are directly subject to box-constraints.
    Forward dynamics constraints are then added as equality constriants to the dynamics.
    """

    q = tc.create_expression(
        "q", "state", (robot.nq, 1)
    )  # joint positions over the trajectory
    q_dot = tc.create_expression("q_dot", "state", (robot.ndof, 1))  # joint velocities

    if options["forward_dynamics_constraints"]:
        tau = tc.create_expression("tau", "control", (robot.ndof, 1))
        q_ddot = robot.fd(q, q_dot, tau)
    else:
        q_ddot = tc.create_expression("q_ddot", "control", (robot.ndof, 1))
        tau = robot.id(q, q_dot, q_ddot)

    # expressions for initial joint position and joint velocity
    q0 = tc.create_expression("q0", "parameter", (robot.ndof, 1))
    q_dot0 = tc.create_expression("q_dot0", "parameter", (robot.ndof, 1))

    tc.set_dynamics(q, q_dot)
    tc.set_dynamics(q_dot, q_ddot)
    print("Adding joint torque constriaints")
    print(robot.joint_torque_ub)
    # add joint position, velocity and acceleration limits
    pos_limits = {
        "lub": True,
        "hard": True,
        "expression": q,
        "upper_limits": robot.joint_ub,
        "lower_limits": robot.joint_lb,
    }
    vel_limits = {
        "lub": True,
        "hard": True,
        "expression": q_dot,
        "upper_limits": robot.joint_vel_ub,
        "lower_limits": robot.joint_vel_lb,
    }
    torque_limits = {
        "lub": True,
        "hard": True,
        "expression": tau,
        "upper_limits": robot.joint_torque_ub,
        "lower_limits": robot.joint_torque_lb,
    }
    joint_constraints = {"path_constraints": [pos_limits, vel_limits, torque_limits]}
    tc.add_task_constraint(joint_constraints)

    # adding the initial constraints on joint position and velocity
    joint_init_con = {"expression": q, "reference": q0}
    joint_vel_init_con = {"expression": q_dot, "reference": q_dot0}
    init_constraints = {"initial_constraints": [joint_init_con, joint_vel_init_con]}
    tc.add_task_constraint(init_constraints)

    return q, q_dot, q_ddot, tau, q0, q_dot0
