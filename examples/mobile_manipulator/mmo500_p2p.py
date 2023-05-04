## OCP for point-to-point motion and visualization of a MMO-500 Mobile Manipulator

from time import sleep
from tasho.templates.P2P import P2P
from tasho import TaskModel, default_mpc_options
from tasho.templates.Regularization import Regularization
from tasho.Variable import Variable
from robotshyu import Robot as rob
from tasho.OCPGenerator import OCPGenerator
import numpy as np
import casadi as cs
import tasho
from tasho.ConstraintExpression import ConstraintExpression
from tasho.Expression import Expression
from tasho.MPC import MPC 
from tasho import environment as env

robot = rob.Robot("mmo_500_ppr")
link_name = 9
goal_pose = Variable(robot.name, "goal_pose", "magic_number", (4,4), np.array(
        [[0, 1, 0, 0.6], [1, 0, 0, 0.0], [0, 0, -1, 0.25], [0, 0, 0, 1]]
    ))
q0 = [1.0193752249977548, -0.05311582280659044, -2.815452580946695,
    1.3191046402052224, 2.8582660722530533, 1.3988994390898029,1.8226311094569714]
robot.set_joint_acceleration_limits(lb=-360*3.14159/180, ub=360*3.14159/180)
ndof = robot.nq
q_current = Variable(robot.name, "jointpos_init", 'magic_number', (ndof, 1), q0)