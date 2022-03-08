import sys
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
from tasho import environment as env
import casadi as cs
from casadi import pi, cos, sin
import tasho.utils.geometry as geometry
import numpy as np
import matplotlib.pyplot as plt

print("Moving-object picking with Kinova Gen3")

################################################
# Define robot and initial joint angles
################################################
# Import the robot object from the robot's repository (includes functions for FD, ID, FK, joint limits, etc)
robot = rob.Robot("kinova", analytical_derivatives=False)

# Update robot's parameters if needed
max_joint_acc = 90 * 3.14159 / 180
robot.set_joint_acceleration_limits(lb=-max_joint_acc, ub=max_joint_acc)

# Define initial conditions of the robot
q_init = [0, -0.523598, 0, 2.51799, 0, -0.523598, -1.5708]
q_dot_init = [0] * robot.ndof

################################################
# Task spacification - Approximation to object
################################################

# Select prediction horizon and sample time for the MPC execution
horizon_size = 10
t_mpc = 0.1

# Initialize the task context object
tc = tp.task_context(horizon_size * t_mpc, horizon_steps = horizon_size)

# Define the input type of the robot (torque or acceleration)
q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(tc, robot)

# Add object of interest for the robot (in this case a cube)
cube_pos = tc.create_expression("cube_pos", "parameter", (3, 1))
T_goal = cs.vertcat(
    cs.hcat([0, 1, 0, cube_pos[0]]),
    cs.hcat([1, 0, 0, cube_pos[1]]),
    cs.hcat([0, 0, -1, cube_pos[2]]),
    cs.hcat([0, 0, 0, 1]),
)

# Define constraints at the end of the horizon (final ee position and final joint velocity)
T_ee = robot.fk(q)[7]

position_con = {"hard": False, "expression": T_ee[0:3,3], "reference": T_goal[0:3,3], "norm": "L1"}
rot_err, _ = geometry.rotmat_to_axisangle(T_ee[0:3, 0:3]@T_goal[0:3, 0:3])
rotation_con = {"hard":False, "expression":rot_err, "reference":0, "norm":"L1"} 
zero_vel = {"hard": True, "expression": q_dot, "reference": 0}
final_constraints = {"final_constraints": [position_con, rotation_con, zero_vel]}
tc.add_task_constraint(final_constraints)

# Add penality terms on joint velocity and acceleration for regulatization

# vel_regularization = {
#     "hard": False,
#     "expression": q_dot,
#     "reference": 0,
#     "gain": 1e-3,
# }
# acc_regularization = {
#     "hard": False,
#     "expression": q_ddot,
#     "reference": 0,
#     "gain": 1e-3,
# }
# task_objective = {"path_constraints": [vel_regularization, acc_regularization]}
# tc.add_task_constraint(task_objective)

tc.add_regularization(
    expression=q_dot, weight=1e-3, norm="L2", variable_type="state", reference=0
)
tc.add_regularization(
    expression=q_ddot, weight=1e-3, norm="L2", variable_type="control", reference=0
)

################################################
# Set solver and discretization options
################################################
# tc.set_ocp_solver("ipopt", {"ipopt": {"print_level": 0,"tol": 1e-3}})
tc.set_ocp_solver("ipopt", {"ipopt": {"print_level": 0,"tol": 1e-3, "linear_solver":"ma27"}}) #use this if you have hsl

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
tc.set_value(cube_pos, [0.5, 0, 0.25])
tc.set_value(q0, q_init)
tc.set_value(q_dot0, q_dot_init)

################################################
# Solve the OCP that describes the task
################################################
sol = tc.solve_ocp()

################################################
# MPC Simulation
################################################
visualizationBullet = True

if visualizationBullet:

    # Create world simulator based on pybullet
    from tasho import WorldSimulator
    import pybullet as p

    obj = WorldSimulator.WorldSimulator()

    # Add robot to the world environment
    position = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0, 1.0]
    kinovaID = obj.add_robot(position, orientation, "kinova")

    # Set environment
    environment = env.Environment()

    cube1 = env.Cube(length = 1, position = [0.5, -0.2, 0.35], orientation = [0.0, 0.0, 0.0, 1.0], urdf = "/models/objects/cube_small.urdf")
    environment.add_object(cube1, "cube")
    
    table1 = env.Box(height = 0.3, position = [0.5, 0, 0], orientation = [0.0, 0.0, 0.7071080798594737, 0.7071054825112364], urdf = "/models/objects/table.urdf")
    environment.add_object(table1, "table1")

    environment.set_in_world_simulator(obj)

    cubeID = environment.get_object_ID("cube")
    p.resetBaseVelocity(cubeID, linearVelocity=[0, 0.8, 0])


    # Determine number of samples that the simulation should be executed
    no_samples = int(t_mpc / obj.physics_ts)
    if no_samples != t_mpc / obj.physics_ts:
        print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

    # Correspondence between joint numbers in bullet and OCP
    joint_indices = [0, 1, 2, 3, 4, 5, 6]

    # Begin the visualization by applying the initial control signal
    ts, q_sol = tc.sol_sample(q, grid="control")
    ts, q_dot_sol = tc.sol_sample(q_dot, grid="control")
    obj.resetJointState(kinovaID, joint_indices, q_init)
    obj.setController(
        kinovaID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
    )

    # Execute the MPC loop
    for i in range(horizon_size * 100):
        print("----------- MPC execution -----------")

        # Predict the position of the target object (cube)
        lin_vel, ang_vel = p.getBaseVelocity(cubeID)
        lin_vel = cs.DM(lin_vel)
        lin_pos, _ = p.getBasePositionAndOrientation(cubeID)
        lin_pos = cs.DM(lin_pos)
        time_to_stop = cs.norm_1(lin_vel) / 0.5
        predicted_pos = (
            cs.DM(lin_pos)
            + cs.DM(lin_vel) * time_to_stop
            - 0.5 * 0.5 * lin_vel / (cs.norm_1(lin_vel) + 1e-3) * time_to_stop ** 2
        )
        predicted_pos[2] += 0.03  # cube height
        print("Predicted position of cube", predicted_pos)

        # Set parameter values
        tc.set_value(q0, q_sol[1])
        tc.set_value(q_dot0, q_dot_sol[1])
        tc.set_value(cube_pos, predicted_pos)

        # Solve the ocp
        sol = tc.solve_ocp()

        # Sample the solution for the next MPC execution
        ts, q_sol = tc.sol_sample(q, grid="control")
        _, q_dot_sol = tc.sol_sample(q_dot, grid="control")
        _, q_ddot_sol = tc.sol_sample(q_ddot)

        tc.set_initial(q, q_sol.T)
        tc.set_initial(q_dot, q_dot_sol.T)
        tc.set_initial(q_ddot, q_ddot_sol.T)

        # Set control signal to the simulated robot
        obj.setController(
            kinovaID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
        )

        # Simulate
        obj.run_simulation(no_samples)

        # Termination criteria
        T_ee_sol = robot.fk(q_sol[0])[7]
        pos_ee_sol = T_ee_sol[:3, 3]
        dist_to_cube_sq = cs.sumsqr(pos_ee_sol - predicted_pos)
        if dist_to_cube_sq <= 2e-2 ** 2:
            break

    obj.run_simulation(100)

    obj.end_simulation()
