import unittest
from tasho import task_prototype_rockit as tp
from tasho import robot as rb
from rockit import MultipleShooting, Ocp
import numpy as np

class TestTask(unittest.TestCase):

    def test_task(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    # def test_sum_tuple(self):
    #     self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

    def test_task_tuple(self):
        tc = tp.task_context(5)

        x = tc.create_expression('x', 'state', (1, 1))

        u = tc.create_expression('u', 'control', (1, 1))

        tc.set_dynamics(x, u)

        task_spec = {'initial_constraints':[{'expression':x, 'reference':0.1}]}
        task_spec['path_constraints'] = [{'expression':u, 'reference':0, 'hard':False, 'gain':1}]
        task_spec['final_constraints'] = [{'expression':x**2, 'reference':5, 'hard':True}]

        tc.add_task_constraint(task_spec)

        solver_options = {"ipopt": {"print_level": 0}, "print_time": False, "expand" : True}

        tc.ocp.solver('ipopt',solver_options)
        tc.ocp.method(MultipleShooting(N = 5, M = 2, intg='rk'))

        ocp = tc.ocp
        sol = ocp.solve()
        t, x_val= sol.sample(x, grid='control')

        tc.generate_function(name = 'opti_o', save=True, codegen=False)

        self.assertAlmostEqual( x_val[-1], 2.236067977499, 10, "Final position test failed")
        self.assertEqual( t[-1], 5, "Final time test failed")

    def test_robotloader(self):
        # Kinova Gen3
        rob_kinova = rb.Robot(name="kinova")

        self.assertEqual(rob_kinova.ndof, 7, "Kinova Gen3 - should have 7 degrees of freedom")

        rob_kinova.set_joint_limits([-3.14,-2,-3.14,-2,-3.14,-2,-3.14],[3.14,2,3.14,2,3.14,2,3.14])

        arr_fromrobot = rob_kinova.fk([0,0,0,0,0,0,0])[rob_kinova.ndof].full()
        arr_expected = np.array([[1, 0, 0, 6.1995e-05],[0,  1,  0, -2.48444537e-02],[0, 0, 1, 1.18738514],[0, 0, 0, 1]])
        self.assertTrue(np.linalg.norm(arr_fromrobot - arr_expected) < 1e-8, "Kinova Gen3 - forward kinematics assert failed")

        # ABB Yumi
        rob_yumi = rb.Robot(name="yumi")

        self.assertEqual(rob_yumi.ndof, 18, "ABB Yumi - should have 18 degrees of freedom")

if __name__ == '__main__':
    unittest.main()
