from time import sleep
from examples.templates.P2P import P2P
from examples.templates.Regularization import Regularization
from tasho.Variable import Variable
from tasho.robot import Robot as rob
from tasho.OCPGenerator import OCPGenerator
import numpy as np
import networkx as nx

case = 1

if case == 1:
    robot = rob("kinova")
    link_name = 7
    goal_pose = Variable(robot.name, "goal_pose", "magic_number", (4,4), np.array(
            [[0, 1, 0, 0.6], [1, 0, 0, 0.], [0, 0, -1, 0.25], [0, 0, 0, 1]]
        ))
    q0 = [1.0193752249977548, -0.05311582280659044, -2.815452580946695,
        1.3191046402052224, 2.8582660722530533, 1.3988994390898029,1.8226311094569714,
    ]
    

robot.set_joint_acceleration_limits(lb=-360*3.14159/180, ub=360*3.14159/180)
ndof = robot.nq
q_current = Variable(robot.name, "jointpos_init", 'magic_number', (ndof, 1), q0)
# q_current = [0]*7


# Using the template to create the P2P task
task_P2P = P2P(robot, link_name, goal_pose, q_current, 0.01)

#Adjusting the regularization for better convergence
reg_jacc = task_P2P.constraint_expressions['reg_qdd_'+robot.name].change_weight(0.1)
task_P2P.add_path_constraints(Regularization(task_P2P.variables['qd_'+robot.name], 1))

# Substituting a variable

horizon_steps = 20
horizon_period = 4
OCP_gen = OCPGenerator(task_P2P, False, {"time_period": horizon_period, "horizon_steps":horizon_steps})
q_ocp = OCP_gen.stage_tasks[0].variables['q_'+robot.name].x
OCP_gen.tc.set_initial(q_ocp, q0)
OCP_gen.tc.solve_ocp()

t_grid, qsol = OCP_gen.tc.sol_sample(q_ocp)
t_grid, q_dot_sol = OCP_gen.tc.sol_sample(OCP_gen.stage_tasks[0].variables['qd_'+robot.name].x)

# Visualization
# from tasho import world_simulator
# import pybullet as p
# obj = world_simulator.world_simulator()
# # Add robot to the world environment
# position = [0.0, 0.0, 0.0]
# orientation = [0.0, 0.0, 0.0, 1.0]
# robotID = obj.add_robot(position, orientation, robot.name)
# joint_indices = [0, 1, 2, 3, 4, 5, 6]
# obj.resetJointState(robotID, joint_indices, q0)
# for i in range(horizon_steps + 1):
#     sleep(horizon_period*0.5/horizon_steps)
#     obj.resetJointState(
#         robotID, joint_indices, qsol[i]
#     )


# sleep(0.5)
# obj.end_simulation()