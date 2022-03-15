from tasho import default_mpc_options, Variable
from tasho.OCPGenerator import OCPGenerator
from tasho.ConstraintExpression import ConstraintExpression
from tasho.Expression import Expression
from tasho.templates.P2P import P2P
from tasho.MPC import MPC 
from tasho import robot as rob
from tasho import environment as env
import casadi as cs
from tasho.templates.Regularization import Regularization
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
q0_val = [0, -0.523598, 0, 2.51799, 0, -0.523598, -1.5708]
qd0_val = [0] * robot.ndof

################################################
# Task spacification - Approximation to object
################################################

# Select prediction horizon and sample time for the MPC execution
horizon_size = 10
t_mpc = 0.05

q_init = Variable.Variable(robot.name, 'q_init', 'parameter', (7,1))
qd_init = Variable.Variable(robot.name, 'qd_init', 'parameter', (7,1))
goal_pose = Variable.Variable(robot.name, "goal_pose", 'parameter', (4,4))

task_P2P = P2P(robot, 7, goal_pose, q_init, 0.001)
task_P2P.remove_initial_constraints(task_P2P.constraint_expressions['stationary_qd_kinova'])
task_P2P.remove_terminal_constraints(
    task_P2P.constraint_expressions['rot_con_pose_7_kinova_vs_goal'], 
    task_P2P.constraint_expressions['trans_con_pose_7_kinova_vs_goal'])
task_P2P.add_initial_constraints(ConstraintExpression(robot.name, "eq", Expression(robot.name, "err_qd_qinit", lambda a, b : a - b, qd_init, task_P2P.variables['qd_kinova']),
                             "hard", reference = 0))
task_P2P.add_path_constraints(Regularization(task_P2P.expressions['trans_error_pose_7_kinova_vs_goal'], 1.0, norm = "L1"),
                            Regularization(task_P2P.expressions['ax_ang_error_pose_7_kinova_vs_goal'], 1), 
                            Regularization(task_P2P.variables['qd_kinova'], 1e-4))

task_P2P.write_task_graph("after_sub.svg")
# Initialize the task context object

pOCP = OCPGenerator(task_P2P, False, {"time_period":horizon_size*t_mpc, "horizon_steps":horizon_size})

tc = pOCP.tc
# tc = tp.task_context(horizon_size * t_mpc, horizon_steps = horizon_size)

# Add object of interest for the robot (in this case a cube)
# cube_pos = tc.create_expression("cube_pos", "parameter", (3, 1))
# T_goal = cs.vertcat(
#     cs.hcat([0, 1, 0, cube_pos[0]]),
#     cs.hcat([1, 0, 0, cube_pos[1]]),
#     cs.hcat([0, 0, -1, cube_pos[2]]),
#     cs.hcat([0, 0, 0, 1]),
# )

# Define constraints at the end of the horizon (final ee position and final joint velocity)


# position_con = {"hard": False, "expression": T_ee[0:3,3], "reference": T_goal[0:3,3], "norm": "L1"}
# rot_err, _ = geometry.rotmat_to_axisangle(T_ee[0:3, 0:3]@T_goal[0:3, 0:3])
# rotation_con = {"hard":False, "expression":rot_err, "reference":0, "norm":"L1"} 
# zero_vel = {"hard": True, "expression": q_dot, "reference": 0}
# final_constraints = {"final_constraints": [position_con, rotation_con, zero_vel]}
# tc.add_task_constraint(final_constraints)

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

# tc.add_regularization(
#     expression=q_dot, weight=1e-3, norm="L2", variable_type="state", reference=0
# )
# tc.add_regularization(
#     expression=q_ddot, weight=1e-3, norm="L2", variable_type="control", reference=0
# )

################################################
# Set solver and discretization options
################################################
# tc.set_ocp_solver("ipopt", {"ipopt": {"print_level": 0,"tol": 1e-3}})
tc.set_ocp_solver("ipopt", {"ipopt": {"print_level": 0,"tol": 1e-3, "linear_solver":"ma27"}}) #use this if you have hsl


################################################
# Set parameter values
################################################
q = pOCP.stage_tasks[0].variables['q_kinova'].x
qd = pOCP.stage_tasks[0].variables['qd_kinova'].x
q_0 = pOCP.stage_tasks[0].variables['q_init_kinova'].x
qd_0 = pOCP.stage_tasks[0].variables['qd_init_kinova'].x
goal_pose = pOCP.stage_tasks[0].variables['goal_pose_kinova'].x
goal_pose_val =  cs.vertcat(
    cs.hcat([0, 1, 0, 0.5]),
    cs.hcat([1, 0, 0, 0.0]),
    cs.hcat([0, 0, -1, 0.25]),
    cs.hcat([0, 0, 0, 1]),
)
tc.set_initial(q, q0_val)
tc.set_value(goal_pose, goal_pose_val)
tc.set_value(q_0, q0_val)
tc.set_value(qd_0, qd0_val)

# Add a monitor for termination criteria
tc.add_monitor({"name":"termination_criteria", "expression":cs.sqrt(cs.sumsqr(pOCP.stage_tasks[0].expressions["trans_error_pose_7_kinova_vs_goal"].x)) - 2e-2, "reference":0, "lower":True, "initial":True})

################################################
# Solve the OCP that describes the task
################################################
sol = tc.solve_ocp()
mpc_options = default_mpc_options.get_default_mpc_options()

# Generate the controller component
tc.ocp_solver = "ipopt"
tc.ocp_options = mpc_options["ipopt_hsl"]
tc.mpc_solver = tc.ocp_solver
tc.mpc_options = tc.ocp_options
vars_db = tc.generate_MPC_component("./", {"ocp_cg_opts":{"save":True, "codegen":False, "jit":False}, "mpc":True, "mpc_cg_opts":{"save":True, "codegen":False, "jit":False}})

MPC_component = MPC("kinova_obj_pickup", "./n" + tc.name + ".json")

################################################
# MPC Simulation
################################################
visualizationBullet = True

if visualizationBullet:

    # Create world simulator based on pybullet
    from tasho import WorldSimulator
    import pybullet as p

    obj = WorldSimulator.WorldSimulator(bullet_gui=True)

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

    

    
    

    cube2 = env.Cube(length = 1, position = [0.5, -0.2, 0.35], orientation = [0.0, 0.0, 0.0, 1.0], urdf = "/models/objects/cube_small copy.urdf", fixed = True)
    environment.add_object(cube2, "cube2")
    environment.set_in_world_simulator(obj)
    cubeID = environment.get_object_ID("cube")
    cube2ID = environment.get_object_ID("cube2")

    p.resetBaseVelocity(cubeID, linearVelocity=[0, 0.8, 0])
    # Determine number of samples that the simulation should be executed
    no_samples = int(t_mpc / obj.physics_ts)
    if no_samples != t_mpc / obj.physics_ts:
        print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

    # Correspondence between joint numbers in bullet and OCP
    joint_indices = [0, 1, 2, 3, 4, 5, 6]

    # Begin the visualization by applying the initial control signal
    ts, q_sol = tc.sol_sample(q, grid="control")
    ts, q_dot_sol = tc.sol_sample(qd, grid="control")
    obj.resetJointState(kinovaID, joint_indices, q0_val)
    obj.setController(
        kinovaID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
    )
    q_log = []
    q_dot_log = []
    predicted_pos_log = []

    # Execute the MPC loop
    for i in range(horizon_size * 100):
        print("----------- MPC execution -----------")

        q_now = obj.readJointPositions(kinovaID, joint_indices)
        qd_now = obj.readJointVelocities(kinovaID, joint_indices)
        
        MPC_component.input_ports["port_inp_q_init_kinova"]["val"] = q_now
        MPC_component.input_ports["port_inp_qd_init_kinova"]["val"] = qd_now

        # Predict the position of the target object (cube)
        lin_vel, ang_vel = p.getBaseVelocity(cubeID)
        lin_vel = cs.DM(lin_vel)
        lin_pos, _ = p.getBasePositionAndOrientation(cubeID)
        lin_pos = cs.DM(lin_pos)
        time_to_stop = cs.norm_1(lin_vel) / 0.3
        predicted_pos = (
            cs.DM(lin_pos)
            + cs.DM(lin_vel) * time_to_stop
            - 0.5 * 0.5 * lin_vel / (cs.norm_1(lin_vel) + 1e-3) * time_to_stop ** 2
        )
        predicted_pos_log.append(predicted_pos.full())
        p.resetBasePositionAndOrientation(cube2ID, predicted_pos.full(), [0.0, 0.0, 0.0, 1.0])
        
        predicted_pos[2] += 0.05  # cube height
        print("Predicted position of cube", predicted_pos)
        # Set parameter values
        
        # tc.set_value(q_0, q_now)
        # tc.set_value(qd_0, q_dot_sol[1])
        predicted_pos_val = cs.vertcat(
            cs.hcat([0, 1, 0, predicted_pos[0]]),
            cs.hcat([1, 0, 0, predicted_pos[1]]),
            cs.hcat([0, 0, -1, predicted_pos[2]]),
            cs.hcat([0, 0, 0, 1]),
        )

        MPC_component.input_ports["port_inp_goal_pose_kinova"]["val"] = cs.vec(predicted_pos_val)
        # tc.set_value(goal_pose, predicted_pos_val)
        
        if i == 0:
            MPC_component.configMPC()

        MPC_component.runMPC()
        # Solve the ocp
        # sol = tc.solve_ocp()

        # Sample the solution for the next MPC execution
        # ts, q_sol = tc.sol_sample(q, grid="control")
        # _, q_dot_sol = tc.sol_sample(qd, grid="control")
        # _, q_ddot_sol = tc.sol_sample(q_ddot)

        q_log.append(q_now)
        q_dot_log.append(qd_now)

        # Set control signal to the simulated robot
        qd_control_sig = MPC_component.output_ports["port_out_qd_kinova"]["val"].full()
        qdd_control_sig = (MPC_component.output_ports["port_out_qdd_kinova"]["val"] * t_mpc).full()
        obj.setController(
            kinovaID, "velocity", joint_indices, targetVelocities=qd_control_sig + qdd_control_sig
        )

        # Simulate
        obj.run_simulation(no_samples)

        # Termination criteria
        if "termination_criteria_true" in MPC_component.event_output_port:
            break

    # # Execute the loop in the end to get a smooth video
    # obj.resetJointState(kinovaID, joint_indices, q0_val)
    # obj.setController(
    #     kinovaID, "velocity", joint_indices, targetVelocities=[0]*7
    # )
    # obj.run_simulation(5000)
    # p.resetBasePositionAndOrientation(cubeID, [0.5, -0.2, 0.35], [0.0, 0.0, 0.0, 1.0])
    # p.resetBaseVelocity(cubeID, linearVelocity=[0, 0.8, 0])
    # for i in range(len(q_dot_sol_log)):
    #     p.resetBasePositionAndOrientation(cube2ID, predicted_pos_log[i], [0.0, 0.0, 0.0, 1.0])
    #     obj.setController(
    #         kinovaID, "velocity", joint_indices, targetVelocities=q_dot_sol_log[i][0]
    #     )
    #     obj.run_simulation(no_samples)
    # obj.setController(
    #     kinovaID, "velocity", joint_indices, targetVelocities=[0]*7
    # )


    
    # obj.run_simulation(500)

    obj.end_simulation()
    # print(predicted_pos_log)
