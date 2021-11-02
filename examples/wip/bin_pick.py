# A tasho example file to prototype the multi-stage algorithm

# First stage, stay above and outside the bin. So the bin is a box from which collision should be avoided.

# Second stage, reach the object upto a specified distance.

# Third stage: Start the approach motion from the specified distance


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
import time

if __name__ == "__main__":

    print("Bin pick motion skill:")

    visualizationBullet = False
    horizon_size = 10
    t_mpc = 0.5
    max_joint_acc = 30 * 3.14159 / 180
    max_joint_vel = 30 * 3.14159 / 180
    time_optimal = False
    horizon_period = 10 #in seconds

    robot = rob.Robot("ur10")

    robot.set_joint_acceleration_limits(lb=-max_joint_acc, ub=max_joint_acc)
    robot.set_joint_velocity_limits(lb=-max_joint_vel, ub=max_joint_vel)


    if time_optimal:
        tc = tp.task_context(horizon_steps = horizon_size)
    else:
        tc = tp.task_context(time= horizon_period, horizon_steps = horizon_size)


    q1, q_dot1, q_ddot1, q_init1, q_dot_init1 = input_resolution.acceleration_resolved(tc, robot, {}, stage = 0)

    fk_vals1 = robot.fk(q1)[6]
    T_goal = np.array(
        [
            [-1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, -0.6],
            [0.0, 0.0, -1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    final_vel = {"hard": True, "expression": q_dot1, "reference": 0}
    final_pose = {"hard": True, "expression": fk_vals1, "reference": T_goal}
    tc.add_task_constraint({"final_constraints":[final_vel, final_pose]}, stage = 0)

    tc.add_regularization(q_dot1, 1e-3)

    tc.set_value(q_init1, [0]*6, stage = 0)
    tc.set_value(q_dot_init1, [0]*6, stage = 0)
    tc.set_discretization_settings({})
    tc.set_ocp_solver("ipopt", {"ipopt":{"linear_solver":"ma27"}})

    #creating the second stage
    stage2 = tc.create_stage(time = 5, horizon_steps = 10)
    q2, q_dot2, q_ddot2 = input_resolution.acceleration_resolved(tc, robot, {'init_parameter':False}, stage = 1)

    tc.ocp.subject_to(tc.stages[0].at_tf(q1) == stage2.at_t0(q2))
    tc.ocp.subject_to(tc.stages[0].at_tf(q_dot1) == stage2.at_t0(q_dot2))
    # tc.set_value(q_init2, [0]*6, stage = 1)
    # tc.set_value(q_dot_init2, [0]*6, stage = 1)

    sol = tc.solve_ocp()

    t_grid, q1_sol = tc.sol_sample(q1, stage = 0)
    t_grid, q2_sol = tc.sol_sample(q2, stage = 1)

    print("q1_sol", q1_sol)
    print("q2_sol", q2_sol)


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
    cg_opts = {"ocp_cg_opts":{"jit":False, "save":True, "codegen":False}}
    varsdb = tc.generate_MPC_component("/home/ajay/Desktop/bin_picking_", cg_opts)

    #print json in a prettified form
    import pprint
    pprint.pprint(varsdb)

    #Load casadi function to test:
    casfun = cs.Function.load("/home/ajay/Desktop/bin_picking_tc_ocp.casadi")

    sol_cg = casfun([0]*1054)
