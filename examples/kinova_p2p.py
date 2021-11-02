## OCP for point-to-point motion of a kinova robot arm

import sys
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("Task specification and visualization of P2P OCP")

    horizon_size = 20
    t_mpc = 0.1
    max_joint_acc = 10
    max_joint_vel = 60*3.14159*180

    robot = rob.Robot("kinova")

    robot.set_joint_acceleration_limits(lb=-max_joint_acc, ub=max_joint_acc)
    robot.set_joint_velocity_limits(lb=-max_joint_vel, ub=max_joint_vel)

    tc = tp.task_context(horizon_size * t_mpc, horizon_steps = horizon_size)

    # q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot, {})
    q, q_dot, q_ddot, tau, q0, q_dot0 = input_resolution.torque_resolved(tc, robot, {"forward_dynamics_constraints":True})

    # computing the expression for the final frame
    fk_vals = robot.fk(q)[7]

    final_pos = {
        "hard": True,
        "expression": fk_vals[0:3, 3],
        "reference": [0.5, 0.0, 0.5],
    }
    # final_vel = {"hard": True, "expression": q_dot, "reference": 0}
    # final_constraints = {"final_constraints": [final_pos, final_vel]}
    # tc.add_task_constraint(final_constraints)

    # adding penality terms on joint velocity and position
    pos_cost = {
        "hard": False,
        "expression": fk_vals[0:3, 3],
        "reference": [0.5, 0.0, 0.5],
        "gain": 1
    }
    vel_regularization = {"hard": False, "expression": q_dot, "reference": 0, "gain": 1e-2}
    acc_regularization = {
        "hard": False,
        "expression": q_ddot,
        "reference": 0,
        "gain": 1e-2,
    }

    # task_objective = {"path_constraints": [pos_cost, vel_regularization, acc_regularization]}
    # tc.add_task_constraint(task_objective)

    tc.stages[0].add_objective(tc.stages[0].sum(cs.sumsqr(q_dot)*1e-2))
    tc.stages[0].add_objective(tc.stages[0].sum(cs.sumsqr(q_ddot)*1e-2))
    # tc.stages[0].add_objective(tc.stages[0].sum(cs.sumsqr(q - [1]*7)))
    tc.stages[0].add_objective(tc.stages[0].sum(cs.sumsqr(fk_vals[0:3, 3] - cs.DM([0.5, 0.0, 0.5]))*1))
    tc.set_ocp_solver("ipopt", {"expand":True, "ipopt":{"linear_solver":"mumps"}})
    tc.set_value(q0, [0] * 7)
    tc.set_value(q_dot0, [0] * 7)
    tc.set_initial(tau, robot.id([0]*7, [0]*7, [0]*7))
    disc_settings = {
        "discretization method": "multiple shooting",
        "order": 1,
        "integration": "rk",
    }
    tc.set_discretization_settings(disc_settings)
    sol = tc.solve_ocp()

    ts, q_sol = tc.sol_sample(q, grid="control")
    print(q_sol)
    print(robot.fk(q_sol[-1, :])[7])

    # print(tc.ocp._method.opti.x)
    # print(tc.ocp._method.opti.lam_g)
    # print(tc.ocp._method.opti.p)
    # print(tc.get_states)
    # print(tc.get_output_states())
    # print(tc.states["q"])
