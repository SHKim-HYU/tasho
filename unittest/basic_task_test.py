import unittest
from tasho import task_prototype_rockit as tp
from rockit import MultipleShooting, Ocp

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

        self.assertTrue( x_val[-1] - 2.236067977 <= 1e-8, "Final position test failed")
        self.assertTrue( t[-1] - 5 <= 1e-5, "Final time test failed")

if __name__ == '__main__':
    unittest.main()
