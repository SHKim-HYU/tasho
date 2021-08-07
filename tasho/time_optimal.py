# Takes a joint-space path of a robot and returns a time optimal
# trajectory. Can choose to optimize with either joint accelerations
# or torques as the control inputs

import casadi as cs
import numpy as np
from tasho import task_prototype_rockit as tp
from rockit import MultipleShooting, FreeGrid, DirectCollocation, SingleShooting

class Time_optimal:

    """
    Class for creating an object that computes time-optimal trajectories for
    a given set of way-points in the joint space
    """

    def __init__(self, robot, no_waypoints, bounded, control_rate_con = None):

        """
            Class constructor - initializes an object for computing time optimal
            trajectories for a given sequence of way-points in the joint space.

            :param robot: The robot model for which the time optimal motions are computed.
            :type robot: TaSHo robot type

            :param no_waypoints: Number of waypoints that will be given in the joint space.
            :type no_waypoints: int

            :param bounded: 'acceleration' for acceleration bound or 'torque' for torque bound.
            :type bounded: string

            :param control_rate_con: (optional) None by default. Otherwise specifies the bound on the rate of change of the control action.
            :type control_rate_con: float
        """

        self.no_waypoints = no_waypoints
        tc = tp.task_context(horizon_steps = no_waypoints) # Creating the task context

        no_joints = robot.ndof

        # Creating parameters for the sequence of joint-space waypoints
        q = tc.create_parameter('q_waypoints', (no_joints, 1), grid = 'control')
        q_dot = tc.create_state('qd', (no_joints, 1)) #joint velocities
        q_acc = tc.create_control('control', (no_joints, 1)) #joint accelerations

        dt = tc.stages[0].next(tc.stages[0].t) - tc.stages[0].t #time step between OCP samples

        # Add bounds on joint velocities
        tc.add_task_constraint({"path_constraints":[{'lub':True, 'hard':True, 'expression':q_dot, 'lower_limits': robot.joint_vel_lb, 'upper_limits': robot.joint_vel_ub, 'include_first':True}]})

        # Add the dynamics constraints
        tc.stages[0].subject_to(tc.stages[0].next(q) == q + dt*q_dot + 0.5*dt**2*q_acc)
        tc.set_dynamics(q_dot, q_acc)


        if bounded == 'acceleration':

            control = q_acc
            # Add bounds on the joint accelerations
            tc.add_task_constraint({"path_constraints":[{'lub':True, 'hard':True, 'expression':control, 'lower_limits': robot.joint_acc_lb, 'upper_limits': robot.joint_acc_ub, 'include_first':True}]})

        elif bounded == 'torque':

            # Compute joint torques
            control = robot.id(q, q_dot, q_acc)

            # Add bounds on the joint torques
            print(robot.joint_torque_lb)
            print(control.shape)
            tc.add_task_constraint({"path_constraints":[{'lub':True, 'hard':True, 'expression':control, 'lower_limits': robot.joint_torque_lb, 'upper_limits': robot.joint_torque_ub, 'include_first':True}]})

        else:

            raise Exception("Invalid bounded option: " + bounded)


        if control_rate_con != None:

            tc.add_task_constraint({"path_constraints":[{'lub':True, 'hard':True, 'expression':(control - tc.stages[0].next(control))/dt, 'lower_limits': -control_rate_con, 'upper_limits': control_rate_con}]})
            tc.stages[0].subject_to(-control_rate_con <= (tc.stages[0].at_t0(control/dt) <= control_rate_con))


        # Add initial and final joint velocity constraints
        tc.add_task_constraint({'initial_constraints':[{"expression":q_dot, "reference": 0, "equality":True, "hard":True}]})
        tc.add_task_constraint({'final_constraints':[{"expression":q_dot, "reference": 0, "equality":True, "hard":True}]})

        self._tc = tc
        self._q = q
        self._q_dot = q_dot
        self._q_acc = q_acc
        self._control = control

        # tc.add_regularization(control, weight = 1e-4)#, norm = 'L1')
        tc.stages[0].add_objective(tc.stages[0].T)
        tc.stages[0].method(MultipleShooting(N=no_waypoints, grid=FreeGrid(localize_T=True), intg='rk'))
        tc.set_ocp_solver('ipopt', {'ipopt':{'linear_solver':'mumps'}})

    def use_ma27(self):

        """
        Set the ocp solver to MA27 is HSL solvers are enabled for ipopt.
        """

        self._tc.set_ocp_solver('ipopt', {'ipopt':{'linear_solver':'ma27'}})

    def compute_time_opt_traj(self, q, q_dot = None, control = None):

        """
            Function to compute the time optimal trajectory for a given sequence of waypoints.

            :param q: A 2D array with joint positions along rows and waypoint sequence running through columns.
            :type q: numpy or CasADi DM 2D array

            :param q_dot: (optional) An initial guess for joint velocities in the same format as the positions.
            :type q_dot: numpy or CasADi DM 2D array

            :param control: (optional) An initial guess for the control actions (either joint torques or accelerations) in the same format as the positions.
            :type control: numpy or CasADi DM 2D array

            Returns optimal time, joint velocity and joint acceleration sequence.
        """

        self._tc.set_value(self._q, q)

        if not q_dot is None:
            self._tc.set_initial(self._q_dot, q_dot.T)

        if not control is None:
            self._tc.set_initial(self._control, control.T)

        # if t_sol.any() != None:
        #     self._tc.set_initial(self._tc.stages[0].t, t_sol)


        sol = self._tc.solve_ocp()

        tsol, asol = self._tc.sol_sample(self._q_acc, grid = 'control')
        qdotsol = self._tc.sol_sample(self._q_dot, grid = 'control')[1]

        return tsol, qdotsol, asol


# if __name__ == '__main__':
#
#     print("No syntax errors")
#     from tasho import robot as rob
#     robot = rob.Robot("iiwa7")
#     joint_acc_limit = 3 #rad/s^2
#     robot.joint_acc_ub = joint_acc_limit
#     robot.joint_acc_lb = -joint_acc_limit
#     topt = Time_optimal(robot, 19, 'acceleration', control_rate_con = None)
#     Ts = 0.2
#     q_val = cs.DM.zeros(7, 19)
#     for i in range(1,10):
#         q_val[:, i] = q_val[:, 0] + 0.5*(Ts*i)**2*joint_acc_limit
#     q_dot = 9*Ts*joint_acc_limit
#     for i in range(10, 19):
#         q_val[:, i] = q_val[:, 9] + q_dot*(i-9)*Ts - 0.5*(Ts*(i-9))**2*joint_acc_limit
#
#     # topt._tc.set_value(topt._q, q_val)
#
#     # topt._tc.set_ocp_solver('ipopt', {'ipopt':{'linear_solver':'ma27'}})
#     # sol = topt._tc.solve_ocp()
#     #
#     #
#     #
#     # tsol, asol = topt._tc.sol_sample(topt._control, grid = 'control')
#     # qdotsol = topt._tc.sol_sample(topt._q_dot, grid = 'control')[1]
#
#     topt.use_ma27()
#     tsol, qdotsol, asol = topt.compute_time_opt_traj(q_val)
#
#     # tsol, asol = topt._tc.sol_sample(topt._control[0], grid = 'control')
#     # qdotsol = topt._tc.sol_sample(topt._q_dot[0], grid = 'control')[1]
#
#     q_verify = [0]
#
#     for i in range(18):
#         dt = tsol[i+1] - tsol[i]
#         q_verify.append(q_verify[-1] + qdotsol[i]*dt + 0.5*dt**2*asol[i])
#
#     print(topt._tc.sol_sample(topt._control[0], grid = 'control'))
#     print(topt._tc.sol_sample(topt._q_dot[0], grid = 'control')[1])
#     print(topt._tc.sol_sample(topt._q[0], grid = 'control')[1])
#     print(q_verify)
