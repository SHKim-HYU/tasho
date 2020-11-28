import unittest
from tasho import task_prototype_rockit as tp

class TestTask(unittest.TestCase):

    def test_add_regularization(self):
        tc = tp.task_context(2)

        x = tc.create_expression('x', 'state', (1, 1))
        u = tc.create_expression('u', 'control', (1, 1))
        tc.set_dynamics(x, u)

        task_spec = {'initial_constraints':[{'expression':x, 'reference':0.05}]}
        tc.add_task_constraint(task_spec)

        tc.add_regularization(x, 10, reference=5)
        tc.add_regularization(u, 1)

        solver_options = {"ipopt": {"print_level": 0}, "print_time": False, "expand" : False}
        tc.set_ocp_solver('ipopt',solver_options)
        disc_settings = {'discretization method': 'multiple shooting', 'horizon size': 10, 'order':2, 'integration':'rk'}
        tc.set_discretization_settings(disc_settings)


        sol = tc.solve_ocp()
        t, x_val= sol.sample(x, grid='control')

        #test the result with L2 regularization
        self.assertAlmostEqual( x_val[-1], 4.98411537, 5, "Final position test failed")
        self.assertEqual( t[-1], 2, "Final time test failed")

        #test the result with high L1 regularization on control.  no motion
        tc.add_regularization(u, 200, norm = 'L1')
        tc.set_ocp_solver('ipopt',solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        t, x_val= sol.sample(x, grid='control')
        self.assertAlmostEqual( x_val[-1], 0.05, 6, "Final position test failed")

        #Testing regularizations on variables
        y = tc.create_expression('y', 'variable', (1, 1))
        tc.add_regularization(y, 10, reference = 5)
        tc.add_regularization(y, 5, reference = 0)
        tc.set_ocp_solver('ipopt',solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = sol.value(y)
        print(y_val)
        self.assertAlmostEqual( y_val, 3.333333333, 6, "Variable regularization failed")

        tc.add_regularization(y, 5, reference = 0)
        tc.set_ocp_solver('ipopt',solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = sol.value(y)
        self.assertAlmostEqual( y_val, 2.500000000000, 6, "Variable regularization failed")

        tc.add_regularization(y, 101, reference = 0, norm = 'L1')
        tc.set_ocp_solver('ipopt',solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = sol.value(y)
        self.assertAlmostEqual( y_val, 0, 6, "Variable regularization failed")

if __name__ == '__main__':
    unittest.main()
