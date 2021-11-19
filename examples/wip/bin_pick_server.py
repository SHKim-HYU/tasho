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
import copy

if __name__ == "__main__":

    print("Bin pick motion skill:")

    visualizationBullet = False
    horizon_size = 10
    max_joint_acc_val = 100 * 3.14159 / 180
    max_joint_vel_val = 40 * 3.14159 / 180
    time_optimal = True
    horizon_period = 2 #in seconds
    pi = 3.14159
    q0_start = [0.0, -pi/3, pi/3, -pi/2, -pi/2, 0.0]
    q0_start = cs.DM([-0.55953752222222, -0.92502372222222, 1.1693696111111, -1.6929679444444, -1.570795, 0])
    obstacle_clearance = 0.03

    obs_cyl = {'radius':0.15, 'z_max': 0.4, 'center':[0.95*10, 0.00], 'z_min':0.0}
    cyl_bottom_val = [obs_cyl['center'][0], obs_cyl['center'][1], obs_cyl['z_min']]
    cyl_top_val = [obs_cyl['center'][0], obs_cyl['center'][1], obs_cyl['z_max']]
    end_effector_height = 0.3
    approach_distance_val = 0.075
    box1 = {'b_height':0.3, 'b_max_x':1.0, 'b_max_y':-0.2, 'b_min_x':0.4, 'b_min_y':-0.8}
    box2 = {'b_height':0.3, 'b_max_x':1.0, 'b_max_y':0.8, 'b_min_x':0.4, 'b_min_y':0.2}

    box_start = box1
    box_dest = box2

    robot = rob.Robot("ur10")



    if time_optimal:
        tc = tp.task_context(horizon_steps = horizon_size)
    else:
        tc = tp.task_context(time= horizon_period, horizon_steps = horizon_size)

    obs_cyl_zmax = tc.create_parameter('obs_cyl_zmax_1', (1,1), stage = 0)
    obs_cyl_zmin = tc.create_parameter('obs_cyl_zmin_1', (1,1), stage = 0)
    obs_cyl_center = tc.create_parameter('obs_cyl_center_1', (2,1), stage = 0)
    obs_cyl_radius = tc.create_parameter('obs_cyl_radius_1', (1,1), stage = 0)
    tc.set_value(obs_cyl_zmax, cyl_top_val[2], stage = 0)
    tc.set_value(obs_cyl_zmin, cyl_bottom_val[2], stage = 0)
    tc.set_value(obs_cyl_radius, obs_cyl['radius'], stage = 0)
    tc.set_value(obs_cyl_center, obs_cyl['center'], stage = 0)
    cyl_bottom = cs.vertcat(obs_cyl_center, obs_cyl_zmin)
    cyl_top = cs.vertcat(obs_cyl_center, obs_cyl_zmax)

    max_joint_acc1 = tc.create_parameter('max_jacc_1', (1,1), stage = 0)
    max_joint_vel1 = tc.create_parameter('max_jvel_1', (1,1), stage = 0)
    robot.set_joint_acceleration_limits(lb=-max_joint_acc1, ub=max_joint_acc1)
    robot.set_joint_velocity_limits(lb=-max_joint_vel1, ub=max_joint_vel1)

    tc.set_value(max_joint_acc1, max_joint_acc_val, stage = 0)
    tc.set_value(max_joint_vel1, max_joint_vel_val, stage = 0)


    q1, q_dot1, q_ddot1, q_init1, q_dot_init1 = input_resolution.acceleration_resolved(tc, robot, {}, stage = 0)

    # Computing the FK
    fk_vals1 = robot.fk(q1)[6]
    fk_ee = fk_vals1
    # Adding the gripper height to get the tip position
    fk_ee[0:3, 3] += fk_vals1[0:3, 0:3]@cs.DM([end_effector_height, 0, 0])
    fk_ee_fun = cs.Function('fk_ee', [q1], [fk_ee])

    T_goal = np.array(
        [
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 1.0, 0.0, 0.6],
            [-1.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # final_vel = {"hard": True, "expression": q_dot1, "reference": 0}
    final_pose_z = {"hard": True, "inequality":True, "expression": -fk_ee[2,3], "upper_limits": -box_dest['b_height']}
    final_pose_x = {"hard":True, "lub":True, "expression":fk_ee[0,3], "upper_limits":box_dest["b_max_x"], "lower_limits":box_dest["b_min_x"]}
    final_pose_y = {"hard":True, "lub":True, "expression":fk_ee[1,3], "upper_limits":box_dest["b_max_y"], "lower_limits":box_dest["b_min_y"]}
    tc.add_task_constraint({"final_constraints":[final_pose_x, final_pose_y, final_pose_z]}, stage = 0)

    #Adding obstacle avoidance constraints
    sep_hyp1 = tc.create_control('sep_hyp', (4,1))
    # sep_hyp1 = cs.vertcat(sep_hyp1, cs.sqrt(1 - cs.sumsqr(sep_hyp1[1:3])))
    obs_to_sep1 = sep_hyp1[0] + sep_hyp1[1:].T@cyl_bottom
    obs_to_sep2 = sep_hyp1[0] + sep_hyp1[1:].T@cyl_top
    ee_to_sep = sep_hyp1[0] + sep_hyp1[1:].T@fk_ee[0:3,3]
    flange_to_sep = sep_hyp1[0] + sep_hyp1[1:].T@fk_vals1[0:3,3]

    #separating hyperplane unit norm
    sh1_unit_norm = {"hard":True, "inequality":True, "expression":cs.sumsqr(sep_hyp1[1:]), "upper_limits":1.0, "include_first":True}
    sh1_con_obs = {"hard":True, "inequality":True, "expression":cs.vertcat(obs_to_sep1, obs_to_sep2), "upper_limits":-obs_cyl_radius - obstacle_clearance, "include_first":True}
    sh1_con_rob = {"hard":True, "inequality":True, "expression":-cs.vertcat(ee_to_sep, flange_to_sep), "upper_limits": - obstacle_clearance, "include_first":True}
    tc.add_task_constraint({"path_constraints":[sh1_unit_norm, sh1_con_obs, sh1_con_rob]})

    tc.add_regularization(q_dot1, 1e-3)
    tc.minimize_time(10, 0)
    tc.set_value(q_init1, q0_start, stage = 0)
    tc.set_value(q_dot_init1, [0]*6, stage = 0)
    tc.set_discretization_settings({})
    tc.set_ocp_solver("ipopt", {"ipopt":{"linear_solver":"ma27"}})

    #creating the second stage: reaching the desired approach motion start point
    if not time_optimal:
        stage2 = tc.create_stage(time = 2, horizon_steps = 5)
    else:
        stage2 = tc.create_stage(horizon_steps = 5)

    max_joint_acc2 = tc.create_parameter('max_jacc_2', (1,1), stage = 1)
    max_joint_vel2 = tc.create_parameter('max_jvel_2', (1,1), stage = 1)
    robot.set_joint_acceleration_limits(lb=-max_joint_acc2, ub=max_joint_acc2)
    robot.set_joint_velocity_limits(lb=-max_joint_vel2, ub=max_joint_vel2)

    tc.set_value(max_joint_acc2, max_joint_acc_val, stage = 1)
    tc.set_value(max_joint_vel2, max_joint_vel_val, stage = 1)

    approach_distance = tc.create_parameter('approach_distance', (1,1), stage = 1)
    tc.set_value(approach_distance, 0.075, stage = 1)

    q2, q_dot2, q_ddot2 = input_resolution.acceleration_resolved(tc, robot, {'init_parameter':False}, stage = 1)


    fk_ee2 = fk_ee_fun(q2)
    T_goal_approach = cs.MX(copy.deepcopy(T_goal))
    final_pose_trans = {"hard": True, "expression": fk_ee2[0:3,3], "reference": T_goal_approach[0:3,3] - T_goal_approach[0:3,0]*approach_distance}
    #strictly enforce the direction of the axis. LICQ fails at the solution. So adding as inequality constraint for robustness
    final_pose_rot = {"hard":True, "inequality":True, "expression":-fk_ee2[0:3,0].T@T_goal_approach[0:3,0], "upper_limits":-0.99}
    tc.add_task_constraint({"final_constraints":[final_pose_trans, final_pose_rot]}, stage = 1)

    tc.ocp.subject_to(tc.stages[0].at_tf(q1) == stage2.at_t0(q2))
    tc.ocp.subject_to(tc.stages[0].at_tf(q_dot1) == stage2.at_t0(q_dot2))
    tc.add_regularization(q_dot2, 1e-3, stage = 1)
    tc.minimize_time(10, 1)

    # creating the third stage: reaching the desired position
    if not time_optimal:
        stage3 = tc.create_stage(time = 0.5, horizon_steps = 5)
    else:
        stage3 = tc.create_stage(horizon_steps = 5)

    max_joint_acc3 = tc.create_parameter('max_jacc_3', (1,1), stage = 2)
    max_joint_vel3 = tc.create_parameter('max_jvel_3', (1,1), stage = 2)
    robot.set_joint_acceleration_limits(lb=-max_joint_acc3, ub=max_joint_acc3)
    robot.set_joint_velocity_limits(lb=-max_joint_vel3, ub=max_joint_vel3)

    tc.set_value(max_joint_acc3, max_joint_acc_val, stage = 2)
    tc.set_value(max_joint_vel3, max_joint_vel_val, stage = 2)

    q3, q_dot3, q_ddot3 = input_resolution.acceleration_resolved(tc, robot, {'init_parameter':False}, stage = 2)
    fk_ee3 = fk_ee_fun(q3)

    # Pose constraints
    final_pose_trans = {"hard": True, "expression": fk_ee3[0:3,3], "reference": T_goal[0:3,3]}
    #strictly enforce the direction of the axis. LICQ fails at the solution. So adding as inequality constraint for robustness
    final_pose_rot = {"hard":True, "inequality":True, "expression":-fk_ee3[0:3,0].T@T_goal_approach[0:3,0], "upper_limits":-0.99}

    #terminal zero velocity constraint
    final_vel_con = {"hard":True, "expression":q_dot2, "reference":0.0}
    tc.add_task_constraint({"final_constraints":[final_pose_trans, final_pose_rot]}, stage = 2)

    tc.add_regularization(q_dot3, 1e-3, stage = 2)
    tc.ocp.subject_to(stage2.at_tf(q2) == stage3.at_t0(q3))
    tc.ocp.subject_to(stage2.at_tf(q_dot2) == stage3.at_t0(q_dot3))
    tc.minimize_time(10, 2)
    # import pdb; pdb.set_trace()
    try:
        tc.set_initial(q1, q0_start)
        tc.set_initial(q2, q0_start, stage=1)
        tc.set_initial(q3, q0_start, stage=2)
        sol = tc.solve_ocp()
    except:
        print("Caught exception, trying thrice with random initializations")
        counter = 0
        while counter < 3:
            q0_randstart = np.random.rand(6,1) - 0.5
            tc.set_initial(q1, q0_randstart)
            tc.set_initial(q2, q0_randstart, stage=1)
            tc.set_initial(q3, q0_randstart, stage=2)
            try:
                sol = tc.solve_ocp()
                break;
            except:
                counter += 1


    t_grid1, q1_sol = tc.sol_sample(q1, stage = 0)
    t_grid2, q2_sol = tc.sol_sample(q2, stage = 1)
    t_grid3, q3_sol = tc.sol_sample(q3, stage = 2)

    print("q1_sol", q1_sol)
    print("q2_sol", q2_sol)
    print("q3_sol", q3_sol)
    print("Time grids", t_grid1, t_grid2+t_grid1[-1], t_grid3 + t_grid2[-1] + t_grid1[-1])
    print("Final pose", fk_ee_fun(q3_sol[-1]))

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
    varsdb = tc.generate_MPC_component("bin_picking_", cg_opts)

    #print json in a prettified form
    import pprint
    pprint.pprint(varsdb)

    #Load casadi function to test:
    casfun = cs.Function.load("bin_picking_tc_ocp.casadi")

    # sol_cg = casfun([0]*1168)












    # implementing the drop off skill

    if time_optimal:
        tc2 = tp.task_context(horizon_steps = horizon_size)
    else:
        tc2 = tp.task_context(time= 0.5, horizon_steps = horizon_size)

    max_joint_acc1 = tc2.create_parameter('max_jacc_1', (1,1), stage = 0)
    max_joint_vel1 = tc2.create_parameter('max_jvel_1', (1,1), stage = 0)
    robot.set_joint_acceleration_limits(lb=-max_joint_acc1, ub=max_joint_acc1)
    robot.set_joint_velocity_limits(lb=-max_joint_vel1, ub=max_joint_vel1)

    tc2.set_value(max_joint_acc1, max_joint_acc_val, stage = 0)
    tc2.set_value(max_joint_vel1, max_joint_vel_val, stage = 0)

    q1, q_dot1, q_ddot1, q_init1, q_dot_init1 = input_resolution.acceleration_resolved(tc2, robot, {}, stage = 0)

    fk_ee = fk_ee_fun(q1)

    final_pose_z = {"hard": True, "inequality":True, "expression": -fk_ee[2,3], "upper_limits": -box_dest['b_height'] - obstacle_clearance}
    final_pose_x = {"hard":True, "lub":True, "expression":fk_ee[0,3], "upper_limits":box_dest["b_max_x"], "lower_limits":box_dest["b_min_x"]}
    final_pose_y = {"hard":True, "lub":True, "expression":fk_ee[1,3], "upper_limits":box_dest["b_max_y"], "lower_limits":box_dest["b_min_y"]}
    tc2.add_task_constraint({"final_constraints":[final_pose_x, final_pose_y, final_pose_z]}, stage = 0)

    tc2.add_regularization(q_dot1, 1e-3)
    tc2.minimize_time(10, 0)
    tc2.set_value(q_init1, q3_sol[-1], stage = 0)
    tc2.set_value(q_dot_init1, [0]*6, stage = 0)
    tc2.set_discretization_settings({})
    tc2.set_ocp_solver("ipopt", {"ipopt":{"linear_solver":"ma27"}})


    #creating the second stage: dropping off the payload
    if not time_optimal:
        stage2 = tc2.create_stage(time = 2, horizon_steps = 5)
    else:
        stage2 = tc2.create_stage(horizon_steps = 5)

    max_joint_acc2 = tc2.create_parameter('max_jacc_2', (1,1), stage = 1)
    max_joint_vel2 = tc2.create_parameter('max_jvel_2', (1,1), stage = 1)
    robot.set_joint_acceleration_limits(lb=-max_joint_acc2, ub=max_joint_acc2)
    robot.set_joint_velocity_limits(lb=-max_joint_vel2, ub=max_joint_vel2)

    tc2.set_value(max_joint_acc2, max_joint_acc_val, stage = 1)
    tc2.set_value(max_joint_vel2, max_joint_vel_val, stage = 1)

    q2, q_dot2, q_ddot2 = input_resolution.acceleration_resolved(tc2, robot, {'init_parameter':False}, stage = 1)

    obs_cyl_zmax = tc2.create_parameter('obs_cyl_zmax_2', (1,1), stage = 1)
    obs_cyl_zmin = tc2.create_parameter('obs_cyl_zmin_2', (1,1), stage = 1)
    obs_cyl_center = tc2.create_parameter('obs_cyl_center_2', (2,1), stage = 1)
    obs_cyl_radius = tc2.create_parameter('obs_cyl_radius_2', (1,1), stage = 1)
    tc2.set_value(obs_cyl_zmax, cyl_top_val[2], stage = 1)
    tc2.set_value(obs_cyl_zmin, cyl_bottom_val[2], stage = 1)
    tc2.set_value(obs_cyl_radius, obs_cyl['radius'], stage = 1)
    tc2.set_value(obs_cyl_center, obs_cyl['center'], stage = 1)
    cyl_bottom = cs.vertcat(obs_cyl_center, obs_cyl_zmin)
    cyl_top = cs.vertcat(obs_cyl_center, obs_cyl_zmax)

    fk_vals2 = robot.fk(q2)[6]
    fk_ee2 = fk_ee_fun(q2)
    #Adding obstacle avoidance constraints
    sep_hyp1 = tc2.create_control('sep_hyp', (4,1), stage=1)
    # sep_hyp1 = cs.vertcat(sep_hyp1, cs.sqrt(1 - cs.sumsqr(sep_hyp1[1:3])))
    obs_to_sep1 = sep_hyp1[0] + sep_hyp1[1:].T@cyl_bottom
    obs_to_sep2 = sep_hyp1[0] + sep_hyp1[1:].T@cyl_top
    ee_to_sep = sep_hyp1[0] + sep_hyp1[1:].T@fk_ee2[0:3,3]
    flange_to_sep = sep_hyp1[0] + sep_hyp1[1:].T@fk_vals2[0:3,3]




    #separating hyperplane unit norm
    sh1_unit_norm = {"hard":True, "inequality":True, "expression":cs.sumsqr(sep_hyp1[1:]), "upper_limits":1.0, "include_first":True}
    sh1_con_obs = {"hard":True, "inequality":True, "expression":cs.vertcat(obs_to_sep1, obs_to_sep2), "upper_limits":-obs_cyl_radius - obstacle_clearance, "include_first":True}
    sh1_con_rob = {"hard":True, "inequality":True, "expression":-cs.vertcat(ee_to_sep, flange_to_sep), "upper_limits": - obstacle_clearance, "include_first":True}
    tc2.add_task_constraint({"path_constraints":[sh1_unit_norm, sh1_con_obs, sh1_con_rob]}, stage = 1)

    final_pose_z = {"hard": True, "inequality":True, "expression": -fk_ee2[2,3], "upper_limits": -box_start['b_height'] - obstacle_clearance}
    final_pose_x = {"hard":True, "lub":True, "expression":fk_ee2[0,3], "upper_limits":box_start["b_max_x"] - obstacle_clearance, "lower_limits":box_start["b_min_x"] + obstacle_clearance}
    final_pose_y = {"hard":True, "lub":True, "expression":fk_ee2[1,3], "upper_limits":box_start["b_max_y"] - obstacle_clearance, "lower_limits":box_start["b_min_y"] + obstacle_clearance}
    final_vel_con = {"hard":True, "expression":q_dot2, "reference":0.0}
    tc2.add_task_constraint({"final_constraints":[final_pose_x, final_pose_y, final_pose_z, final_vel_con]}, stage = 1)

    tc2.add_regularization(q_dot2, 1e-3, stage = 1)
    tc2.ocp.subject_to(tc2.stages[0].at_tf(q1) == stage2.at_t0(q2))
    tc2.ocp.subject_to(tc2.stages[0].at_tf(q_dot1) == stage2.at_t0(q_dot2))
    tc2.minimize_time(10, 1)

    q0_start = q3_sol[-1]
    # sol = tc2.solve_ocp()

    try:
        tc2.set_initial(q1, q0_start)
        tc2.set_initial(q2, q0_start, stage=1)
        sol = tc2.solve_ocp()

    except:
        print("Caught exception, trying thrice with random initializations")
        counter = 0
        while counter < 3:
            q0_randstart = np.random.rand(6,1) - 0.5
            tc2.set_initial(q1, q0_randstart)
            tc2.set_initial(q2, q0_randstart, stage=1)
            try:
                sol = tc2.solve_ocp()
                break;
            except:
                counter += 1

    t_grid1, q1_sol = tc2.sol_sample(q1, stage = 0)
    t_grid2, q2_sol = tc2.sol_sample(q2, stage = 1)

    print("q1_sol", q1_sol)
    print("q2_sol", q2_sol)
    print("Time grids", t_grid1, t_grid2+t_grid1[-1])
    print("Final pose", fk_ee_fun(q2_sol[-1]))

    tc2.add_monitor(
    {
        "name": "termination_criteria",
        "expression": cs.sqrt(cs.sumsqr(q_dot1)) - 0.001,
        "reference": 0.0,  # doesn't matter
        "greater": True,  # doesn't matter
        "initial": True,
    }
)

## Performing code-generation (to potentially deploy an orocos component)
varsdb = tc2.generate_MPC_component("bin_dropping_", cg_opts)

#print json in a prettified form
import pprint
pprint.pprint(varsdb)

#Load casadi function to test:
casfun2 = cs.Function.load("bin_dropping_tc_ocp.casadi")

# sol_cg = casfun2([0]*858)
