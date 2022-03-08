import casadi as cs
from time import sleep
from tasho.templates.P2P import P2P
from tasho.templates.SE3_tunnel import SE3Tunnel
from tasho.templates.Regularization import Regularization
from tasho.Variable import Variable
from robotsmeco import Robot as rob
from tasho.OCPGenerator import OCPGenerator
import numpy as np


# Use the SIMULATE variable to enable simulation on PyBullet
SIMULATE = False

vel_limit = 0.5 #m/s
acc_limit = 2.0 #m/s^2

trans_tunnel_size = 0.02
rot_tunnel_size = 5/3.14159/180

case = 1

if case == 1:
    robot = rob.Robot("kinova")
    link_name = 7
    a_p = 0.2
    z_p = 0.1
    SE3_path_fun = lambda s : cs.vertcat(
                        cs.horzcat(cs.DM([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), 
                            cs.vertcat(0.6+z_p*cs.sin(s*(4*np.pi)), 0.0+a_p*cs.sin(s*(2*np.pi)), 0.25+a_p*cs.sin(s*(2*np.pi))*cs.cos(s*(2*np.pi)))), 
                        cs.DM([0, 0, 0, 1]).T)
    goal_pose = Variable(robot.name, "goal_pose", "magic_number", (4,4), np.array(
            [[-1, 0, 0, 0.6], [0, 1, 0, 0.35], [0, 0, -1, 0.25], [0, 0, 0, 1]]
        ))
    q0 = [ 0.42280387,  1.56128753, -2.07387664,  1.1543891,   1.7809308,   2.03112421,
   4.02677039]


tunnel_task = SE3Tunnel("contouring", SE3_path_fun, vel_limit, acc_limit, trans_tunnel_size, rot_tunnel_size)
    
robot.set_joint_acceleration_limits(lb=-360*3.14159/180, ub=360*3.14159/180)
ndof = robot.nq
q_current = Variable(robot.name, "jointpos_init", 'magic_number', (ndof, 1), q0)

# # Using the template to create the P2P task
task_P2P = P2P(robot, link_name, goal_pose, q_current, 0.001)
# Removing the goal pose of P2P because not relevant for tunnel-following
task_P2P.remove_expression(goal_pose)

# Substituting the SE3_traj in tunnel-following with the fk_pose of the robot
tunnel_task.substitute_expression(tunnel_task.variables['SE3_traj_contouring'], task_P2P.expressions["pose_7_kinova"])

# Including task_P2P within to create the tunnel-following task
tunnel_task.include_subtask(task_P2P)

# Uncomment the following line to generate the task graph
# tunnel_task.write_task_graph("tunnel_following.svg")
# task_P2P.write_task_graph("task_tunnel_p2p.svg")


# Substituting a variable
horizon_steps = 30
horizon_period = 3
OCP_gen = OCPGenerator(tunnel_task, False, {"time_period": horizon_period, "horizon_steps":horizon_steps})
q_ocp = OCP_gen.stage_tasks[0].variables['q_'+robot.name].x
OCP_gen.tc.set_initial(q_ocp, q0)

OCP_gen.tc.set_ocp_solver(
    "ipopt", 
    # {"ipopt":{"linear_solver":"ma27"}} 
)

OCP_gen.tc.solve_ocp()

st = OCP_gen.stage_tasks[0]
print(OCP_gen.tc.sol_sample(st.expressions['SE3_path_contouring'].x))
t_grid, qsol = OCP_gen.tc.sol_sample(q_ocp)
# print(qsol)
t_grid, q_dot_sol = OCP_gen.tc.sol_sample(OCP_gen.stage_tasks[0].variables['qd_'+robot.name].x)


if SIMULATE:
    # Visualization
    from tasho import WorldSimulator
    import pybullet as p
    obj = WorldSimulator.WorldSimulator()

    # Add robot to the world environment
    position = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0, 1.0]
    robotID = obj.add_robot(position, orientation, robot.name)
    joint_indices = [0, 1, 2, 3, 4, 5, 6]

    obj.resetJointState(robotID, joint_indices, q0)

    for i in range(horizon_steps + 1):
        sleep(horizon_period*0.5/horizon_steps)
        obj.resetJointState(
            robotID, joint_indices, qsol[i]
        )

    sleep(0.5)
    obj.end_simulation()