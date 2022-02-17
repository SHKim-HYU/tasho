from time import sleep
from examples.templates.BoxConstraint import BoxConstraint
from examples.templates.P2P import P2P
from examples.templates.Regularization import Regularization
from tasho.ConstraintExpression import ConstraintExpression
from tasho.Expression import Expression
from tasho.Variable import Variable
from robotsmeco import Robot as rob
from tasho.OCPGenerator import OCPGenerator
import numpy as np
import networkx as nx

import logging

logging._Level = 'info'

robot = rob.Robot("franka_panda")
# robot.set_joint_acceleration_limits(lb = -5, ub = 5)
link_name = 7

goal_pose = Variable(robot.name, "goal_pose", "parameter", (4,4))
    
q0 = [-2.41764669e-01,  2.46839298e-01,  2.56913581e-01, -1.96144913e+00,
  -8.14570796e-02,  2.19022196e+00, -2.29644958e+00]

qdd_lb = Variable(robot.name, "qdd_lb", "parameter", (7,1))
qdd_ub = Variable(robot.name, "qdd_ub", "parameter", (7,1))
qd_ub = Variable(robot.name, "qd_ub", "parameter", (7,1))
qd_lb = Variable(robot.name, "qd_lb", "parameter", (7,1))
qddd_ub = Variable(robot.name, "qddd_ub", "parameter", (7,1))
qddd_lb = Variable(robot.name, "qddd_lb", "parameter", (7,1))

joint_jerk_limits = np.array([50]*7)

# qdd = Variable(robot.name, "qdd", "state", (7,1))

q_current = Variable(robot.name, "q_init", "parameter", (7,1))

task_P2P = P2P(robot, link_name, goal_pose, q_current, 0.001)
q = task_P2P.variables['q_'+robot.name]
qdd = task_P2P.variables['qdd_'+robot.name]
qd = task_P2P.variables['qd_'+robot.name]
task_P2P.remove_constraint_expression(task_P2P.constraint_expressions['limits_qd_'+robot.name])
task_P2P.remove_constraint_expression(task_P2P.constraint_expressions['limits_qdd_'+robot.name])
qddd = task_P2P.create_variable(robot.name, "qddd", "control", (7,1))
qdd.type = "state"
task_P2P.set_der(qdd, qddd)



task_P2P.add_path_constraints(BoxConstraint(qddd, qddd_lb, qddd_ub), 
                            BoxConstraint(qdd, qdd_lb, qdd_ub),
                            BoxConstraint(qd, qd_lb, qd_ub))
task_P2P.write_task_graph("task_graph.svg")
# # access the states of the P2P task
# print(task_P2P.constraints)


####################################################################
reg_jacc = task_P2P.constraint_expressions['reg_qdd_'+robot.name].change_weight(0.1)
task_P2P.add_path_constraints(Regularization(task_P2P.variables['qddd_'+robot.name], 0.01))
####################################################################

# Substituting a variable

horizon_steps = 50
horizon_period = 2
OCP = OCPGenerator(task_P2P, True, {"time_period": horizon_period, "horizon_steps":horizon_steps})
q_ocp = OCP.stage_tasks[0].variables['q_'+robot.name].x

variables = OCP.stage_tasks[0].variables
tc = OCP.tc

#################################################################
tc.set_initial(q_ocp, q0)
tc.set_initial(OCP.tc.stages[0].T, 10)
tc.set_value(variables["qd_lb_franka_panda"].x, -3)
tc.set_value(variables["qd_ub_franka_panda"].x, 3)
tc.set_value(variables["qdd_lb_franka_panda"].x, -10)
tc.set_value(variables["qdd_ub_franka_panda"].x, 10)
tc.set_value(variables["qddd_lb_franka_panda"].x, -30)
tc.set_value(variables["qddd_ub_franka_panda"].x, 30)
tc.set_value(variables['q_init_'+robot.name].x, q0)
tc.set_value(variables['goal_pose_'+robot.name].x, np.array(
            [[-1, 0, 0, 0.6], [0, 1, 0, 0.35], [0, 0, -1, 0.25], [0, 0, 0, 1]]
        ))
###################################################################
tc.set_ocp_solver("ipopt", {"ipopt":{"linear_solver":"ma27"}})
tc.minimize_time(100)
tc.solve_ocp()

t_grid, qsol = OCP.tc.sol_sample(q_ocp)
print(t_grid)
t_grid, q_dot_sol = OCP.tc.sol_sample(OCP.stage_tasks[0].variables['qd_'+robot.name].x)
horizon_period = t_grid[-1]

sampler_q = tc.sol(tc.stages[0]).sampler('q_traj_fun', variables['q_franka_panda'].x)
sampler_qd = tc.sol(tc.stages[0]).sampler('qd_traj_fun', variables['qd_franka_panda'].x)
t_sampling = np.arange(0, t_grid[-1], 0.001)

sampler_q = tc.sol(tc.stages[0]).sampler('q_traj_fun', variables['q_franka_panda'].x)
sampler_qd = tc.sol(tc.stages[0]).sampler('qd_traj_fun', variables['q_franka_panda'].x)

q_sampling = [sampler_q(x).full().flatten() for x in t_sampling]
qd_sampling = [sampler_qd(x).full().flatten() for x in t_sampling]
import matplotlib.pyplot as plt

plt.figure()
plt.plot(t_sampling, q_sampling)

plt.figure()
plt.plot(t_sampling, qd_sampling)

# plt.show()

# Visualization
from tasho import world_simulator
import pybullet as p
obj = world_simulator.world_simulator()
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