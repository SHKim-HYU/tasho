from tasho.templates.P2P import P2P
from tasho.templates.MobileManipulator import MoMa
from tasho import TaskModel, default_mpc_options
from tasho.templates.Regularization import Regularization
from tasho.Variable import Variable
from tasho.ConstraintExpression import ConstraintExpression
from tasho.Expression import Expression
from tasho.MPC import MPC 
from tasho import environment as env
from tasho import WorldSimulator
from tasho.OCPGenerator import OCPGenerator
from robotshyu import Robot as rob
from time import sleep
import numpy as np
import casadi as cs
import pybullet as p
import tasho
import os

# Options for example codes
frame_enable = True
robot_name = "hyumm_scan"
robot = rob.Robot(robot_name)
link_name = 6

# Define initial conditions of the robot
q0_val = [0, 0, 0, 0, -0.423598, -0.9, 0.0, -1.5709, -0.523598]
qd0_val = [0] * robot.ndof

robot.set_joint_acceleration_limits(lb=-360*3.14159/180, ub=360*3.14159/180)

################################################
# Task spacification - Approximation to object
################################################

# Select prediction horizon and sample time for the MPC execution
horizon_size = 10
t_mpc = 0.05

q_init = Variable(robot.name, 'q_init', 'parameter', (robot.ndof,1))
qd_init = Variable(robot.name, 'qd_init', 'parameter', (robot.ndof,1))
goal_pose = Variable(robot.name, "goal_pose", 'parameter', (4,4))

task_MoMa = MoMa(robot, link_name, goal_pose, q_init, rot_tol=1e-3)
#task_MoMa.write_task_graph("hyumm_MoMa_Task4.svg")

task_MoMa.remove_initial_constraints(task_MoMa.constraint_expressions['stationary_qd_' + robot_name])
task_MoMa.remove_terminal_constraints('rot_con_pose_'+str(link_name)+ '_' + robot_name + '_vs_goal',
                                    'trans_con_pose_'+str(link_name)+ '_' + robot_name + '_vs_goal')
#task_MoMa.write_task_graph("hyumm_MoMa_Task5.svg")

task_MoMa.add_initial_constraints(ConstraintExpression(robot.name, "eq", Expression(robot.name, "err_qd_qinit", lambda a, b : a - b, qd_init, task_MoMa.variables['qd_'+robot_name]),
                             "hard", reference = 0))
task_MoMa.add_path_constraints(Regularization(task_MoMa.expressions['trans_error_pose_'+str(link_name)+'_' + robot_name + '_vs_goal'], 5e2, norm = "L2"),
                            Regularization(task_MoMa.expressions['ax_ang_error_pose_'+str(link_name)+'_' + robot_name + '_vs_goal'], 1e1), 
                            Regularization(task_MoMa.variables['qd_'+robot_name], 1e0),
                            Regularization(task_MoMa.variables['qdd_'+robot_name], 8e-3))

#task_MoMa.write_task_graph("hyumm_MoMa_Task6.svg")

pOCP = OCPGenerator(task_MoMa, False, {"time_period":horizon_size*t_mpc, "horizon_steps":horizon_size})

tc = pOCP.tc

################################################
# Set solver and discretization options
################################################
mpc_options = default_mpc_options.get_default_mpc_options()

tc.ocp_solver = mpc_options["ipopt_init"]["solver_name"]
tc.ocp_options = mpc_options["ipopt_init"]["options"]
tc.mpc_solver = mpc_options["ipopt_lbfgs"]["solver_name"]
tc.mpc_options =mpc_options["ipopt_lbfgs"]["options"]
tc.set_ocp_solver(tc.ocp_solver, tc.ocp_options)

################################################
# Set parameter values
################################################
q = pOCP.stage_tasks[0].variables['q_'+robot_name].x
qd = pOCP.stage_tasks[0].variables['qd_'+robot_name].x
q_0 = pOCP.stage_tasks[0].variables['q_init_'+robot_name].x
qd_0 = pOCP.stage_tasks[0].variables['qd_init_'+robot_name].x
goal_pose = pOCP.stage_tasks[0].variables['goal_pose_'+robot_name].x
goal_pose_val =  cs.vertcat(
    cs.hcat([-1, 0, 0, 0.75]),
    cs.hcat([0, 1, 0, 0.0]),
    cs.hcat([0, 0, -1, 0.25]),
    cs.hcat([0, 0, 0, 1]),
)
tc.set_initial(q, q0_val)
tc.set_value(goal_pose, goal_pose_val)
tc.set_value(q_0, q0_val)
tc.set_value(qd_0, qd0_val)

# Add an output port for joint velocities as well
tc.tc_dict["out_ports"].append({"name":"port_out_qd_"+robot_name, "var":"qd_"+robot_name, "desc": "output port for the joint velocities"})

# Add a monitor for termination criteria
tc.add_monitor({"name":"termination_criteria", "expression":cs.sqrt(cs.sumsqr(pOCP.stage_tasks[0].expressions["trans_error_pose_"+str(link_name)+"_" + robot_name + "_vs_goal"].x)) - 2e-2, "reference":0, "lower":True, "initial":True})

# os.system("export OMP_NUM_THREADS = 1")

sol = tc.solve_ocp()

dir_casadi_func = "casadi_dir"
os.makedirs("./"+dir_casadi_func, exist_ok=True)
vars_db = tc.generate_MPC_component("./"+dir_casadi_func+"/", {"ocp_cg_opts":{"save":True, "codegen":False, "jit":False}, "mpc":True, "mpc_cg_opts":{"save":True, "codegen":False, "jit":False}})

MPC_component = MPC("hyumm_obj_pickup", "./"+dir_casadi_func+"/"+ tc.name + ".json")

obj = WorldSimulator.WorldSimulator(bullet_gui=True)

# Add robot to the world environment
position = [0.0, 0.0, 0.0]
orientation = [0.0, 0.0, 0.0, 1.0]
com_x = robot.CoM_x(q0_val).full().T[0]
robotID = obj.add_robot(position,orientation,robot_pkg=robot)
comID = obj.add_object_urdf(com_x, orientation, "com.urdf", fixedBase=True, robot_pkg=robot)

if frame_enable:
    frameIDs = [0]*horizon_size
    for i in range(horizon_size):
        frameIDs[i] = obj.add_robot(position,orientation,robot_pkg=robot,frame=True)

# Set environment
environment = env.Environment()
package_path = tasho.__path__[0]
cube1 = env.Cube(length = 1, position = [0.75, -0.2, 0.35], orientation = [0.0, 0.0, 0.0, 1.0], urdf = package_path+"/models/objects/cube_small_green.urdf")
environment.add_object(cube1, "cube")
table1 = env.Box(height = 0.3, position = [0.75, 0, 0], orientation = [0.0, 0.0, 0.7071080798594737, 0.7071054825112364], urdf =package_path+ "/models/objects/table.urdf")
environment.add_object(table1, "table1")

environment.set_in_world_simulator(obj)
cubeID = environment.get_object_ID("cube")

# Determine number of samples that the simulation should be executed
no_samples = int(t_mpc / obj.physics_ts)
if no_samples != t_mpc / obj.physics_ts:
    print("[ERROR] MPC sampling time not integer multiple of physics sampling time")

# Correspondence between joint numbers in bullet and OCP
joint_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9]

 # Begin the visualization by applying the initial control signal
ts, q_sol = tc.sol_sample(q, grid="control")
ts, q_dot_sol = tc.sol_sample(qd, grid="control")

obj.resetJointState(robotID,joint_indices,q0_val)
if frame_enable==True:
    obj.resetMultiJointState(frameIDs, joint_indices, [q0_val])

obj.setController(
    robotID, "velocity", joint_indices, targetVelocities=q_dot_sol[0]
)
q_log = []
q_dot_log = []
predicted_pos_log = []
q_pred = [0]*horizon_size

# reset cube velocity
p.resetBaseVelocity(cubeID, linearVelocity=[0, 0.8, 0])
# p.resetBaseVelocity(cubeID, linearVelocity=[-1, 0, 0])
# p.resetBaseVelocity(cubeID, linearVelocity=[1, 0, 0])
identity_orientation = p.getQuaternionFromEuler([0,0,0])
# Execute the MPC loop

for i in range(horizon_size * 100000):
    print("----------- MPC execution -----------")

    q_now = obj.readJointPositions(robotID, joint_indices)
    qd_now = obj.readJointVelocities(robotID, joint_indices)

    com_x = robot.CoM_x(q_now).full().T[0]
    p.resetBasePositionAndOrientation(comID, com_x, identity_orientation)
    MPC_component.input_ports["port_inp_q_init_"+robot_name]["val"] = q_now
    MPC_component.input_ports["port_inp_qd_init_"+robot_name]["val"] = qd_now

    # the position of the target object (cube)
    lin_pos, _ = p.getBasePositionAndOrientation(cubeID)
    lin_pos = cs.DM(lin_pos)

    predicted_pos = lin_pos
    predicted_pos_log.append(predicted_pos.full())
    predicted_pos[2] += 0.06  # cube height
    print("Predicted position of cube", predicted_pos)
    predicted_pos_val = cs.vertcat(
        cs.hcat([-1, 0, 0, predicted_pos[0]]),
        cs.hcat([0, 1, 0, predicted_pos[1]]),
        cs.hcat([0, 0, -1, predicted_pos[2]]),
        cs.hcat([0, 0, 0, 1]),
    )

    MPC_component.input_ports["port_inp_goal_pose_"+robot_name]["val"] = cs.vec(predicted_pos_val)
        
    if i == 0:
        MPC_component.configMPC()

    MPC_component.runMPC()
    if frame_enable==True:
        sol = MPC_component.res_vals
        for i in range(horizon_size):
            q_pred[i]=sol[robot.ndof*i:robot.ndof*i+robot.ndof].full()
        obj.resetMultiJointState(frameIDs, joint_indices, q_pred)

    q_log.append(q_now)
    q_dot_log.append(qd_now)

    # Set control signal to the simulated robot
    qd_control_sig = MPC_component.output_ports["port_out_qd_"+robot_name]["val"].full()
    qdd_control_sig = (MPC_component.output_ports["port_out_qdd_"+robot_name]["val"] * t_mpc).full()
    
    obj.setController(
        robotID, "velocity", joint_indices, targetVelocities=qd_control_sig+qdd_control_sig
    )
    
    # Simulate
    obj.run_simulation(no_samples)

    # Termination criteria
#     if "termination_criteria_true" in MPC_component.event_output_port:
#         break

obj.end_simulation()

max(MPC_component.solver_times)