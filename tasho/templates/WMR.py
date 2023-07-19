from tasho.ConstraintExpression import ConstraintExpression
from tasho.TaskModel import Task
from tasho.Expression import Expression
from tasho.templates.BoxConstraint import BoxConstraint
from tasho.templates.ConstraintSE3 import ConstraintSE3
from tasho.templates.Regularization import Regularization
import casadi as cs

def WMR(robot, tc=None,  options={}, stage = 0):
    """Function returns the expressions for acceleration-resolved control
    with appropriate position, velocity and acceleration constraints added
    to the task context.

    It focus on kinematical base

    :param tc: The task context

    :param robot: robot The object of the robot in question

    :param options: Dictionary to pass further miscellaneous options
    """

    # Nonholonomic
    if "nonholonomic" in options:
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

            tc.set_dynamics(x, v*cs.cos(th), stage = stage)
            tc.set_dynamics(y, v*cs.sin(th), stage = stage)
            tc.set_dynamics(th, w, stage = stage)

            # add joint position, velocity and acceleration limits
            v_limits = {
                "lub": True,
                "hard": True,
                "expression": v,
                "upper_limits": robot.task_vel_ub[0],
                "lower_limits": robot.task_vel_lb[0],
            }
            w_limits = {
                "lub": True,
                "hard": True,
                "expression": w,
                "upper_limits": robot.task_vel_ub[1],
                "lower_limits": robot.task_vel_lb[1],
            }
            task_constraints = {"path_constraints": [v_limits, w_limits]}
            tc.add_task_constraint(task_constraints, stage = stage)

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

            tc.set_dynamics(x, v*cs.cos(th), stage = stage)
            tc.set_dynamics(y, v*cs.sin(th), stage = stage)
            tc.set_dynamics(th, w, stage = stage)
            tc.set_dynamics(v, dv, stage = stage)
            tc.set_dynamics(w, dw, stage = stage)

            # add joint position, velocity and acceleration limits
            v_limits = {
                "lub": True,
                "hard": True,
                "expression": v,
                "upper_limits": robot.task_vel_ub[0],
                "lower_limits": robot.task_vel_lb[0],
            }
            w_limits = {
                "lub": True,
                "hard": True,
                "expression": w,
                "upper_limits": robot.task_vel_ub[1],
                "lower_limits": robot.task_vel_lb[1],
            }
            dv_limits = {
                "lub": True,
                "hard": True,
                "expression": dv,
                "upper_limits": robot.task_acc_ub[0],
                "lower_limits": robot.task_acc_lb[0],
            }
            dw_limits = {
                "lub": True,
                "hard": True,
                "expression": dw,
                "upper_limits": robot.task_acc_ub[1],
                "lower_limits": robot.task_acc_lb[1],
            }
            task_constraints = {"path_constraints": [v_limits, w_limits, dv_limits, dw_limits]}
            tc.add_task_constraint(task_constraints, stage = stage)

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
                dv, dv0 = tc.create_state("dv" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory
                dw, dw0 = tc.create_state("dw" + str(stage), (1, 1), init_parameter = True, stage = stage)  # joint positions over the trajectory

            else:
                x = tc.create_state("x" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
                y = tc.create_state("y" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
                th = tc.create_state("th" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
                v = tc.create_state("v" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
                w = tc.create_state("w" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
                dv = tc.create_state("dv" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
                dw = tc.create_state("dw" + str(stage), (1, 1), init_parameter = False, stage = stage)  # joint positions over the trajectory
                
            # v=[v,w].T
            ddv = tc.create_control("ddv"+ str(stage), (1, 1), stage = stage)
            ddw = tc.create_control("ddw"+ str(stage), (1, 1), stage = stage)

            tc.set_dynamics(x, v*cs.cos(th), stage = stage)
            tc.set_dynamics(y, v*cs.sin(th), stage = stage)
            tc.set_dynamics(th, w, stage = stage)
            tc.set_dynamics(v, dv, stage = stage)
            tc.set_dynamics(w, dw, stage = stage)
            tc.set_dynamics(dv, ddv, stage = stage)
            tc.set_dynamics(dw, ddw, stage = stage)

            # add joint position, velocity and acceleration limits
            v_limits = {
                "lub": True,
                "hard": True,
                "expression": v,
                "upper_limits": robot.task_vel_ub[0],
                "lower_limits": robot.task_vel_lb[0],
            }
            w_limits = {
                "lub": True,
                "hard": True,
                "expression": w,
                "upper_limits": robot.task_vel_ub[1],
                "lower_limits": robot.task_vel_lb[1],
            }
            dv_limits = {
                "lub": True,
                "hard": True,
                "expression": dv,
                "upper_limits": robot.task_acc_ub[0],
                "lower_limits": robot.task_acc_lb[0],
            }
            dw_limits = {
                "lub": True,
                "hard": True,
                "expression": dw,
                "upper_limits": robot.task_acc_ub[1],
                "lower_limits": robot.task_acc_lb[1],
            }
            ddv_limits = {
                "lub": True,
                "hard": True,
                "expression": ddv,
                "upper_limits": robot.task_jerk_ub[0],
                "lower_limits": robot.task_jerk_lb[0],
            }
            ddw_limits = {
                "lub": True,
                "hard": True,
                "expression": ddw,
                "upper_limits": robot.task_jerk_ub[1],
                "lower_limits": robot.task_jerk_lb[1],
            }
            task_constraints = {"path_constraints": [v_limits, w_limits, dv_limits, dw_limits, ddv_limits, ddw_limits]}
            tc.add_task_constraint(task_constraints, stage = stage)

            if init_parameter:
                # adding the initial constraints on joint position and velocity
                x_init_con = {"expression": x, "reference": x0}
                y_init_con = {"expression": y, "reference": y0}
                th_init_con = {"expression": th, "reference": th0}
                v_init_con = {"expression": v, "reference": v0}
                w_init_con = {"expression": w, "reference": w0}
                dv_init_con = {"expression": dv, "reference": dv0}
                dw_init_con = {"expression": dw, "reference": dw0}
                init_constraints = {"initial_constraints": [x_init_con, y_init_con, th_init_con, v_init_con, w_init_con, dv_init_con, dw_init_con]}
                tc.add_task_constraint(init_constraints, stage = stage)

            if init_parameter:
                return x, y, th, v, w, dv, dw, ddv, ddw, x0, y0, th0, v0, w0, dv0, dw0
            else:
                return x, y, th, v, w, dv, dw, ddv, ddw

        else:
            print("ERROR: Not implemented and probably not recommended")
    # Omni-directional
    else:
        wmr = Task(robot.name,"WMR")
        if "velocity" in options:
            # creating the state and control variables
            q = wmr.create_variable(robot.name, "q", "state", (robot.nq, 1))
            dq = wmr.create_variable(robot.name, "dq", "control", (robot.nq, 1))
            

            # setting the derivatives of the position and velocity terms
            wmr.set_der(q, dq)
            
            return wmr

        elif "acceleration" in options:
            # creating the state and control variables
            q = wmr.create_variable(robot.name, "q", "state", (robot.nq, 1))
            dq = wmr.create_variable(robot.name, "dq", "state", (robot.nq, 1))
            ddq = wmr.create_variable(robot.name, "ddq", "control", (robot.nq, 1))

            # setting the derivatives of the position and velocity terms
            wmr.set_der(q, dq)
            wmr.set_der(dq, ddq)

            return wmr

        elif "jerk" in options:
            # creating the state and control variables
            q = wmr.create_variable(robot.name, "q", "state", (robot.nq, 1))
            dq = wmr.create_variable(robot.name, "dq", "state", (robot.nq, 1))
            ddq = wmr.create_variable(robot.name, "ddq", "state", (robot.nq, 1))
            jq = wmr.create_variable(robot.name, "jq", "control", (robot.nq, 1))

            # setting the derivatives of the position and velocity terms
            wmr.set_der(q, dq)
            wmr.set_der(dq, ddq)
            wmr.set_der(ddq, jq)

        else:
            print("ERROR: Not implemented and probably not recommended")