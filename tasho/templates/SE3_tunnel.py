""" 
Implements a tunnel-following task in SE(3). 

"""
from tasho.templates.BoxConstraint import BoxConstraint
from tasho.templates.ConstraintSE3 import ConstraintSE3
from tasho.ConstraintExpression import ConstraintExpression
import casadi as cs
from tasho.templates.Regularization import Regularization
from tasho.TaskModel import Task
from tasho.Expression import Expression

def SE3Tunnel(name, SE3_path_fun, vel_limit, acc_limit, trans_tunnel_size, rot_tunnel_size,
     prog_start = 0, prog_end = 1):

     
    SE3Tunnel = Task(name, "SE3Tunnel")

    s = SE3Tunnel.create_variable(name, "prog_var", "state", (1,1))
    sd = SE3Tunnel.create_variable(name, "prog_vel", "state", (1,1))
    sdd = SE3Tunnel.create_variable(name, "prog_acc", "control", (1,1))

    rot_tunnel_L1penalty = SE3Tunnel.create_variable(name, "rot_tunnel_L1penalty", "magic_number", (1,1), 20)
    trans_tunnel_L1penalty = SE3Tunnel.create_variable(name, "trans_tunnel_L1penalty", "magic_number", (1,1), 20)
    rot_reference_weight = SE3Tunnel.create_variable(name, "rot_weight", "magic_number", (1,1), 5)
    trans_reference_weight = SE3Tunnel.create_variable(name, "trans_weight", "magic_number", (1,1), 5)

    SE3_traj = SE3Tunnel.create_variable(name, "SE3_traj", "control", (4,4))

    SE3Tunnel.set_der(s, sd)
    SE3Tunnel.set_der(sd, sdd)

    SE3_path = Expression(name, "SE3_path", lambda s : SE3_path_fun(s), s) 
    con_trans, con_rot = ConstraintSE3(SE3_traj, SE3_path, rot_tol = 1e-6)

    error_trans = con_trans._expression
    error_rot = con_rot._expression
    error_trans_dist = Expression(error_trans.uid, "sqrt", lambda e : e.T@e, error_trans )
    con_rot = ConstraintExpression(SE3_path.uid + "_vs_" + SE3_traj.uid, "rot_tunnel_con", error_rot, "soft", ub = rot_tunnel_size, norm = 'L1', weight = rot_tunnel_L1penalty)
    con_trans = ConstraintExpression(SE3_path.uid + "_vs_" + SE3_traj.uid, "trans_tunnel_con", error_trans_dist, "soft", ub = trans_tunnel_size**2, norm = 'L1', weight = trans_tunnel_L1penalty)

    # initial constraints: add starting s position, starting sd
    SE3Tunnel.add_initial_constraints(ConstraintExpression(s.uid, "init", s, "hard", reference = prog_start),
                                ConstraintExpression(sd.uid, "init", sd, "hard", reference = 0))


    # path constraints: add s limits, sd_limits, sdd_limits, trans_tunnel, rot_tunnel, trans_error_reg, rot_error_reg
    SE3Tunnel.add_path_constraints(BoxConstraint(s, lb = prog_start, ub = prog_end),
                                BoxConstraint(sd, lb = 0, ub = vel_limit),
                                BoxConstraint(sdd, lb = -acc_limit, ub = acc_limit),
                                con_trans,
                                con_rot,
                                Regularization(error_rot, rot_reference_weight),
                                Regularization(error_trans, trans_reference_weight),
                                Regularization(sdd, 1e-3), 
                                Regularization(sd, 1e-3),
                                Regularization(s, 1e-3),
    )


    # terminal constraints: add final s, final sd
    SE3Tunnel.add_terminal_constraints(
        ConstraintExpression(s.uid, "terminal", s, "hard", reference = prog_end),
                                ConstraintExpression(sd.uid, "terminal", sd, "hard", reference = 0))


    return SE3Tunnel

