import unittest
from tasho import task_prototype_rockit as tp
import casadi as cs
from os import remove

class TestTask(unittest.TestCase):
    # NOTE: Don't execute this test when using CI/CD
    def test_codegen(self):
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

        ocp = tc.ocp
        ocp.set_value(p, 0.9)
        sol = ocp.solve()
        t, x_val= sol.sample(x, grid='control')

        ## Test function generation/save
        tc.generate_function(name = 'opti_o', save=True, codegen=False)

        loaded_func = cs.Function.load('opti_o.casadi')
        final_x = loaded_func(0.9,cs.vertcat(0,0,0,0,0,0,0,0,0,0,0),cs.vertcat(0,0,0,0,0,0,0))[0][-1]
        self.assertAlmostEqual( final_x, x_val[-1], 10, "Function save/load - final position test failed")
        
        remove('opti_o.casadi')



if __name__ == '__main__':
    unittest.main()
