import unittest
from tasho import robot as rb
import numpy as np

class TestTask(unittest.TestCase):

    def test_robotloader(self):
        # Kinova Gen3
        rob_kinova = rb.Robot(name="kinova")

        self.assertEqual(rob_kinova.ndof, 7, "Kinova Gen3 - should have 7 degrees of freedom")

        # rob_kinova.set_joint_limits([-3.14,-2,-3.14,-2,-3.14,-2,-3.14],[3.14,2,3.14,2,3.14,2,3.14])
        rob_kinova.set_joint_limits(1.25,2.12)
        # print(rob_kinova.joint_ub)

        arr_fromrobot = rob_kinova.fk([0,0,0,0,0,0,0])[rob_kinova.ndof].full()
        arr_expected = np.array([[1, 0, 0, 6.1995e-05],[0,  1,  0, -2.48444537e-02],[0, 0, 1, 1.18738514],[0, 0, 0, 1]])
        self.assertTrue(np.linalg.norm(arr_fromrobot - arr_expected) < 1e-8, "Kinova Gen3 - forward kinematics assert failed")

        rob_kinova.set_from_json("kinova.json")

        self.assertEqual(rob_kinova.ndof, 7, "Kinova Gen3 - should have 7 degrees of freedom (from json)")

        # ABB Yumi
        rob_yumi = rb.Robot(name="yumi")

        self.assertEqual(rob_yumi.ndof, 18, "ABB Yumi - should have 18 degrees of freedom")

        rob_yumi.set_from_json("yumi.json")

        self.assertEqual(rob_yumi.ndof, 18, "ABB Yumi - should have 18 degrees of freedom (from json)")


if __name__ == '__main__':
    unittest.main()
