from tasho.ConstraintExpression import ConstraintExpression
from tasho.TaskModel import Task
from tasho.Expression import Expression
from tasho.templates.BoxConstraint import BoxConstraint
from tasho.templates.ConstraintSE3 import ConstraintSE3
from tasho.templates.Regularization import Regularization
import casadi as cs

def MoMa(robot, link_name, goal_pose, current_location, rot_tol = 1e-3):

    """
    Abstract task that returns a jointspace point-to-point motion task object.

    :param robot: the robot on which the P2P motion task must be executed.
    :type robot: Tasho robot object.

    :param link_name: The link number on the robot for which the goal pose is specified. @TODO: change to link name
    :type link_name: int

    :param goal_pose: The desired pose of the specified link of the specified robot.
    :type goal_pose: SE(3) 4X4 matrix

    :param current_location: Current jointspace pose of the robot arm.
    :type current_location: n-dimensional vector.
    """

    p2p = Task(robot.name, "P2P")
        
    # creating the state and control variables
    q = p2p.create_variable(robot.name, "q", "state", (robot.ndof, 1))
    qd = p2p.create_variable(robot.name, "qd", "state", (robot.ndof, 1))
    qdd = p2p.create_variable(robot.name, "qdd", "control", (robot.ndof, 1))

    # setting the derivatives of the position and velocity terms
    p2p.set_der(q, qd)
    p2p.set_der(qd, qdd)

    # Current pose of the specified link
    fk_pose = Expression(robot.name, "pose_"+str(link_name), lambda q : robot.fk(q)[link_name], q)

    joint_pos_residual = Expression(q.uid +"_"+ current_location.uid, "error", lambda q, c : q - c, q, current_location)

    # Add the initial joint position constraint
    p2p.add_initial_constraints(ConstraintExpression(q.uid, "equality", joint_pos_residual, "hard", reference = 0),
                                ConstraintExpression(qd.uid, "stationary", qd, "hard", reference = 0))

    # Adding the joint position, velocity and acceleration limits
    p2p.add_path_constraints(BoxConstraint(q, robot.joint_lb, robot.joint_ub),
                            BoxConstraint(qd, robot.joint_vel_lb, robot.joint_vel_ub),
                            BoxConstraint(qdd, robot.joint_acc_lb, robot.joint_acc_ub))
                            # Regularization(qdd, 1e-3))

    # Pose error between the specified link and the desired link
    p2p.add_terminal_constraints(*ConstraintSE3(fk_pose, goal_pose, rot_tol),
                              ConstraintExpression(qd.uid, "stationary", qd, "hard", reference = 0))
    
    return p2p

