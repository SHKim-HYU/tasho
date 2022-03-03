import unittest
from tasho.TimeOptimal import TimeOptimal
import casadi as cs


class TestTopt(unittest.TestCase):

    from tasho import robot as rob
    robot = rob.Robot("iiwa7")
    joint_acc_limit = 3 #rad/s^2
    robot.joint_acc_ub = joint_acc_limit
    robot.joint_acc_lb = -joint_acc_limit

    Ts = 0.2
    q_val = cs.DM.zeros(7, 19)
    for i in range(1,10):
        q_val[:, i] = q_val[:, 0] + 0.5*(Ts*i)**2*joint_acc_limit
    q_dot = 9*Ts*joint_acc_limit
    for i in range(10, 19):
        q_val[:, i] = q_val[:, 9] + q_dot*(i-9)*Ts - 0.5*(Ts*(i-9))**2*joint_acc_limit

    # self.robot = robot
    # self.q_val = q_val

    def test_acc_bounded(self):

        topt = TimeOptimal(self.robot, 19, 'acceleration', control_rate_con = None)
        tsol, qdotsol, asol = topt.compute_time_opt_traj(self.q_val)

        q_verify = [0]

        for i in range(18):
            dt = tsol[i+1] - tsol[i]
            q_verify.append(q_verify[-1] + qdotsol[i][0]*dt + 0.5*dt**2*asol[i][0])

            self.assertAlmostEqual(q_verify[i], self.q_val[0,i], 8)

        self.assertAlmostEqual(tsol[-1], 3.6, 6)

        #Test with warm starting
        q_verify = [0]
        tsol, qdotsol, asol = topt.compute_time_opt_traj(self.q_val, qdotsol, asol)
        for i in range(18):
            dt = tsol[i+1] - tsol[i]
            q_verify.append(q_verify[-1] + qdotsol[i][0]*dt + 0.5*dt**2*asol[i][0])

            self.assertAlmostEqual(q_verify[i], self.q_val[0,i], 8)

        self.assertAlmostEqual(tsol[-1], 3.6, 6)

    def test_jerk_bounded(self):

        topt = TimeOptimal(self.robot, 19, 'acceleration', control_rate_con = 10)
        tsol, qdotsol, asol = topt.compute_time_opt_traj(self.q_val)

        q_verify = [0]

        for i in range(18):
            dt = tsol[i+1] - tsol[i]
            q_verify.append(q_verify[-1] + qdotsol[i][0]*dt + 0.5*dt**2*asol[i][0])

            self.assertAlmostEqual(q_verify[i], self.q_val[0,i], 6)

        self.assertAlmostEqual(tsol[-1], 3.6672702652137614, 6)

    def test_torque_bounded(self):

        topt = TimeOptimal(self.robot, 19, 'torque')
        tsol, qdotsol, asol = topt.compute_time_opt_traj(self.q_val)

        q_verify = [0]

        for i in range(18):
            dt = tsol[i+1] - tsol[i]
            q_verify.append(q_verify[-1] + qdotsol[i][0]*dt + 0.5*dt**2*asol[i][0])

            self.assertAlmostEqual(q_verify[i], self.q_val[0,i], 6)

        self.assertAlmostEqual(tsol[-1], 1.5317873613057065, 6)

    def test_torque_rate_bounded(self):

        topt = TimeOptimal(self.robot, 19, 'torque', control_rate_con = 200)
        tsol, qdotsol, asol = topt.compute_time_opt_traj(self.q_val)

        q_verify = [0]

        for i in range(18):
            dt = tsol[i+1] - tsol[i]
            q_verify.append(q_verify[-1] + qdotsol[i][0]*dt + 0.5*dt**2*asol[i][0])

            self.assertAlmostEqual(q_verify[i], self.q_val[0,i], 6)

        self.assertAlmostEqual(tsol[-1], 3.3498249075608415, 6)



if __name__ == "__main__":
    unittest.main()
