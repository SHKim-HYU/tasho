import unittest
from tasho import task_prototype_rockit as tp

class TestTask(unittest.TestCase):

    def test_task_hw(self):
        tc = tp.task_context(5)

        x = tc.create_expression('x', 'state', (1, 1))

        u = tc.create_expression('u', 'control', (1, 1))

        p = tc.create_expression('p', 'parameter', (1, 1))

        tc.set_dynamics(x, u)

        task_spec = {'initial_constraints':[{'expression':x, 'reference':0.05}]}
        task_spec['path_constraints'] = [{'expression':u, 'reference':0, 'hard':False, 'gain':1}]
        task_spec['final_constraints'] = [{'expression':x**2, 'reference':p, 'hard':True}]

        tc.add_task_constraint(task_spec)

        solver_options = {"ipopt": {"print_level": 0}, "print_time": False, "expand" : True}

        tc.set_ocp_solver('ipopt',solver_options)
        disc_settings = {'discretization method': 'multiple shooting', 'horizon size': 5, 'order':2, 'integration':'rk'}
        tc.set_discretization_settings(disc_settings)

        tc.ocp.set_value(p, 5)

        sol = tc.solve_ocp()
        t, x_val= sol.sample(x, grid='control')

        self.assertAlmostEqual( x_val[-1], 2.236067977499, 10, "Final position test failed")
        self.assertEqual( t[-1], 5, "Final time test failed")

        ## Test parameter change
        tc.ocp.set_value(p, 0.9)
        sol = tc.solve_ocp()
        t, x_val= sol.sample(x, grid='control')
        self.assertAlmostEqual( x_val[-1], 0.9486832980505, 10, "Final position test failed")
        # self.assertEqual( t[-1], 5, "Final time test failed")

if __name__ == '__main__':
    unittest.main()
