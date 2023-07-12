# helper functions using Tasho to set up the variables and parameters
# for many standard cases such as velocity-resolved, acceleration-resolved and
# Torque-resolved MPCs to simplify code

# import sys
from tasho import task_prototype_rockit as tp
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
import numpy as np


def acceleration_resolved(tc, robot, options={}, stage = 0):

    """Function returns the expressions for acceleration-resolved control
    with appropriate position, velocity and acceleration constraints added
    to the task context.

    :param tc: The task context

    :param robot: robot The object of the robot in question

    :param options: Dictionary to pass further miscellaneous options
    """

    if 'init_parameter' not in options:
        init_parameter = True
    else:
        init_parameter = options['init_parameter']

    if init_parameter:
        q, q0 = tc.create_state(
            "q" + str(stage), (robot.nq, 1), init_parameter = True, stage = stage
        )  # joint positions over the trajectory
        q_dot, q_dot0 = tc.create_state("q_dot"+ str(stage), (robot.ndof, 1), init_parameter = True, stage = stage)  # joint velocities
    else:
        q = tc.create_state(
            "q" + str(stage), (robot.nq, 1), init_parameter = False, stage = stage
        )  # joint positions over the trajectory
        q_dot = tc.create_state("q_dot"+ str(stage), (robot.ndof, 1), init_parameter = False, stage = stage)  # joint velocities

    q_ddot = tc.create_control("q_ddot"+ str(stage), (robot.ndof, 1), stage = stage)

    tc.set_dynamics(q, q_dot, stage = stage)
    tc.set_dynamics(q_dot, q_ddot, stage = stage)

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
    tc.add_task_constraint(joint_constraints, stage = stage)

    if init_parameter:
        # adding the initial constraints on joint position and velocity
        joint_init_con = {"expression": q, "reference": q0}
        joint_vel_init_con = {"expression": q_dot, "reference": q_dot0}
        init_constraints = {"initial_constraints": [joint_init_con, joint_vel_init_con]}
        tc.add_task_constraint(init_constraints, stage = stage)

    if init_parameter:
        return q, q_dot, q_ddot, q0, q_dot0
    else:
        return q, q_dot, q_ddot


def velocity_resolved(tc, robot, options):

    print("ERROR: Not implemented and probably not recommended")


def torque_resolved(tc, robot, options={"forward_dynamics_constraints": False}, vb_ab = False):

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
        if vb_ab:
            tau, vb, ab = robot.id(q, q_dot, q_ddot)
        else:
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
        "include_first":True
    }
    joint_constraints = {"path_constraints": [pos_limits, vel_limits, torque_limits]}
    # tc.add_task_constraint(joint_constraints)

    # adding the initial constraints on joint position and velocity
    joint_init_con = {"expression": q, "reference": q0}
    joint_vel_init_con = {"expression": q_dot, "reference": q_dot0}
    init_constraints = {"initial_constraints": [joint_init_con, joint_vel_init_con]}
    tc.add_task_constraint(init_constraints)

    if vb_ab:
        return q, q_dot, q_ddot, tau, q0, q_dot0, vb, ab
    else:
        return q, q_dot, q_ddot, tau, q0, q_dot0

def nonholonomic_WMR(tc, robot, options={}, stage = 0):
    """Function returns the expressions for acceleration-resolved control
    with appropriate position, velocity and acceleration constraints added
    to the task context.

    It focus on kinematical base

    :param tc: The task context

    :param robot: robot The object of the robot in question

    :param options: Dictionary to pass further miscellaneous options
    """
    if "velocity" in options:

        if 'init_parameter' not in options:
            init_parameter = True
        else:
            init_parameter = options['init_parameter']

        if init_parameter:
            # x=[x,y,theta].T
            x, x0 = tc.create_state("x" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
            y, y0 = tc.create_state("y" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
            th, th0 = tc.create_state("th" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
            
        else:
            x = tc.create_state("x" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
            y = tc.create_state("y" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
            th = tc.create_state("th" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory

        # v=[v,w].T
        v = tc.create_control("v"+ str(stage), (1, 1), stage = stage)
        w = tc.create_control("w"+ str(stage), (1, 1), stage = stage)

        tc.set_dynamics(x, v*cos(th), stage = stage)
        tc.set_dynamics(y, v*sin(th), stage = stage)
        tc.set_dynamics(th, w, stage = stage)

        # # add joint position, velocity and acceleration limits
        # v_limits = {
        #     "lub": True,
        #     "hard": True,
        #     "expression": q_dot,
        #     "upper_limits": robot.joint_vel_ub,
        #     "lower_limits": robot.joint_vel_lb,
        # }
        # w_limits = {
        #     "lub": True,
        #     "hard": True,
        #     "expression": q_ddot,
        #     "upper_limits": robot.joint_acc_ub,
        #     "lower_limits": robot.joint_acc_lb,
        # }
        # joint_constraints = {"path_constraints": [v_limits, w_limits]}
        # tc.add_task_constraint(joint_constraints, stage = stage)

        if init_parameter:
            # adding the initial constraints on joint position and velocity
            x_init_con = {"expression": x, "reference": x0}
            y_init_con = {"expression": y, "reference": y0}
            th_init_con = {"expression": th, "reference": th0}
            init_constraints = {"initial_constraints": [x_init_con, y_init_con, th_init_con]}
            tc.add_task_constraint(init_constraints, stage = stage)

        if init_parameter:
            return x, y, th, v, w, x0, y0, th0
        else:
            return x, y, th, v, w

    elif "acceleration" in options:

        if 'init_parameter' not in options:
            init_parameter = True
        else:
            init_parameter = options['init_parameter']

        if init_parameter:
            # x=[x,y,theta].T
            x, x0 = tc.create_state("x" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
            y, y0 = tc.create_state("y" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
            th, th0 = tc.create_state("th" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
            v, v0 = tc.create_state("v" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
            w, w0 = tc.create_state("w" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
            
        else:
            x = tc.create_state("x" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
            y = tc.create_state("y" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
            th = tc.create_state("th" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
            v = tc.create_state("v" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
            w = tc.create_state("w" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
            
        # v=[v,w].T
        dv = tc.create_control("dv"+ str(stage), (1, 1), stage = stage)
        dw = tc.create_control("dw"+ str(stage), (1, 1), stage = stage)

        tc.set_dynamics(x, v*cos(th), stage = stage)
        tc.set_dynamics(y, v*sin(th), stage = stage)
        tc.set_dynamics(th, w, stage = stage)
        tc.set_dynamics(v, dv, stage = stage)
        tc.set_dynamics(w, dw, stage = stage)

        # # add joint position, velocity and acceleration limits
        # v_limits = {
        #     "lub": True,
        #     "hard": True,
        #     "expression": v,
        #     "upper_limits": robot.joint_vel_ub,
        #     "lower_limits": robot.joint_vel_lb,
        # }
        # w_limits = {
        #     "lub": True,
        #     "hard": True,
        #     "expression": w,
        #     "upper_limits": robot.joint_acc_ub,
        #     "lower_limits": robot.joint_acc_lb,
        # }
        # a_limits = {
        #     "lub": True,
        #     "hard": True,
        #     "expression": a,
        #     "upper_limits": robot.joint_vel_ub,
        #     "lower_limits": robot.joint_vel_lb,
        # }
        # dw_limits = {
        #     "lub": True,
        #     "hard": True,
        #     "expression": dw,
        #     "upper_limits": robot.joint_acc_ub,
        #     "lower_limits": robot.joint_acc_lb,
        # }
        # joint_constraints = {"path_constraints": [v_limits, w_limits, a_limits, dw_limits]}
        # tc.add_task_constraint(joint_constraints, stage = stage)

        if init_parameter:
            # adding the initial constraints on joint position and velocity
            x_init_con = {"expression": x, "reference": x0}
            y_init_con = {"expression": y, "reference": y0}
            th_init_con = {"expression": th, "reference": th0}
            v_init_con = {"expression": v, "reference": v0}
            w_init_con = {"expression": w, "reference": w0}
            init_constraints = {"initial_constraints": [x_init_con, y_init_con, th_init_con, v_init_con, w_init_con]}
            tc.add_task_constraint(init_constraints, stage = stage)

        if init_parameter:
            return x, y, th, v, w, dv, dw, x0, y0, th0, v0, w0
        else:
            return x, y, th, v, w, dv, dw

    elif "jerk" in options:
        print("ERROR: Not implemented and probably not recommended")

    else:
        print("ERROR: Not implemented and probably not recommended")