import unittest
from tasho import task_prototype_rockit as tp
import casadi as cs

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
        tc.add_regularization(y, 10, variable_type = 'variable', reference = 5)
        tc.add_regularization(y, 5, variable_type = 'variable', reference = 0)
        tc.set_ocp_solver('ipopt',solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = sol.value(y)
        self.assertAlmostEqual( y_val, 3.333333333, 6, "Variable regularization failed")

        tc.add_regularization(y, 5, variable_type = 'variable', reference = 0)
        tc.set_ocp_solver('ipopt',solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = sol.value(y)
        self.assertAlmostEqual( y_val, 2.500000000000, 6, "Variable regularization failed")

        tc.add_regularization(y, 101, variable_type = 'variable', reference = 0, norm = 'L1')
        tc.set_ocp_solver('ipopt',solver_options)
        tc.set_discretization_settings(disc_settings)
        sol = tc.solve_ocp()
        y_val = sol.value(y)
        self.assertAlmostEqual( y_val, 0, 6, "Variable regularization failed")

    def test_initial_constraints(self):
        #Testing the addition of the initial constraints
        solver_options = {"ipopt": {"print_level": 0}, "print_time": False, "expand" : False}
        tc = tp.task_context(2)

        x = tc.create_expression('x', 'state', (1, 1))
        u = tc.create_expression('u', 'control', (1, 1))
        tc.set_dynamics(x, u)

        tc.add_task_constraint({'initial_constraints':[{'expression':x, 'reference':0.05}]})
        tc.add_task_constraint({'initial_constraints':[{'expression':u, 'reference':1.0}]})
        tc.set_ocp_solver('ipopt', solver_options)
        tc.set_discretization_settings({'discretization method': 'multiple shooting', 'horizon size': 10, 'order':1, 'integration':'rk'})
        sol = tc.solve_ocp()
        _, x_val= sol.sample(x, grid='control')
        _, u_val= sol.sample(u, grid='control')
        self.assertAlmostEqual(x_val[0], 0.05, 6, "initial constraint on scalar state not respected" )
        self.assertAlmostEqual(u_val[0], 1.0, 6, "initial constraint on scalar control not respected" )

        del tc

        tc = tp.task_context(2)

        x = tc.create_expression('x', 'state', (2, 1))
        u = tc.create_expression('u', 'control', (2, 1))
        tc.set_dynamics(x, u)

        tc.add_task_constraint({'initial_constraints':[{'expression':x, 'reference':[0.05, 0.1]}]})
        tc.add_task_constraint({'initial_constraints':[{'expression':u, 'reference':[1.0, -1.0]}]})
        tc.set_ocp_solver('ipopt', solver_options)
        tc.set_discretization_settings({'discretization method': 'multiple shooting', 'horizon size': 10, 'order':1, 'integration':'rk'})
        sol = tc.solve_ocp()
        _, x_val= sol.sample(x, grid='control')
        _, u_val= sol.sample(u, grid='control')
        self.assertAlmostEqual(x_val[0,0], 0.05, 6, "initial constraint on vector state not respected" )
        self.assertAlmostEqual(x_val[0,1], 0.1, 6, "initial constraint on vector state not respected" )
        self.assertAlmostEqual(u_val[0,0], 1.0, 6, "initial constraint on vector control not respected" )
        self.assertAlmostEqual(u_val[0,1], -1.0, 6, "initial constraint on vector control not respected" )

    def test_final_constraints(self):
        #Testing the addition of the final constraints
        solver_options = {"ipopt": {"print_level": 0}, "print_time": False, "expand" : False}
        #testing hard terminal constraints on scalars
        tc = tp.task_context(2)
        x = tc.create_expression('x', 'state', (1, 1))
        u = tc.create_expression('u', 'control', (1, 1))
        tc.set_dynamics(x, u)

        tc.add_task_constraint({'final_constraints':[{'expression':x, 'reference':0.05, 'hard':True}]})
        tc.add_task_constraint({'final_constraints':[{'expression':u, 'reference':1.0, 'hard':True}]})
        tc.set_ocp_solver('ipopt', solver_options)
        tc.set_discretization_settings({'discretization method': 'multiple shooting', 'horizon size': 10, 'order':1, 'integration':'rk'})
        sol = tc.solve_ocp()
        _, x_val= sol.sample(x, grid='control')
        _, u_val= sol.sample(u, grid='control')
        self.assertAlmostEqual(x_val[-1], 0.05, 6, "terminal constraint on scalar state not respected" )
        self.assertAlmostEqual(u_val[-1], 1.0, 6, "terminal constraint on scalar control not respected" )

        del tc

        #testing hard terminal constraints on vectors
        tc = tp.task_context(2)

        x = tc.create_expression('x', 'state', (2, 1))
        u = tc.create_expression('u', 'control', (2, 1))
        tc.set_dynamics(x, u)
        tc.add_task_constraint({'initial_constraints':[{'expression':x, 'reference':[0.0, 0.0]}]})
        tc.add_task_constraint({'initial_constraints':[{'expression':u, 'reference':[0.0, 0.0]}]})
        tc.add_regularization(x, 10)
        tc.add_regularization(u, 10)

        tc.add_task_constraint({'final_constraints':[{'expression':x, 'reference':[0.05, 0.1], 'hard':True}]})
        tc.add_task_constraint({'final_constraints':[{'expression':u, 'reference':[1.0, -1.0], 'hard':True}]})
        tc.set_ocp_solver('ipopt', solver_options)
        tc.set_discretization_settings({'discretization method': 'multiple shooting', 'horizon size': 10, 'order':1, 'integration':'rk'})
        sol = tc.solve_ocp()
        _, x_val= sol.sample(x, grid='control')
        _, u_val= sol.sample(u, grid='control')
        self.assertAlmostEqual(x_val[-1,0], 0.05, 6, "final constraint on vector state not respected" )
        self.assertAlmostEqual(x_val[-1,1], 0.1, 6, "final constraint on vector state not respected" )
        self.assertAlmostEqual(u_val[-1,0], 1.0, 6, "final constraint on vector control not respected" )
        self.assertAlmostEqual(u_val[-1,1], -1.0, 6, "final constraint on vector control not respected" )

        #testing quadratic terminal costs
        del tc
        tc = tp.task_context(2)
        x = tc.create_expression('x', 'state', (2, 1))
        u = tc.create_expression('u', 'control', (2, 1))
        tc.set_dynamics(x, u)
        tc.add_regularization(x, 10)
        tc.add_regularization(u, 10)
        tc.add_task_constraint({'initial_constraints':[{'expression':x, 'reference':[0.0, 0.0]}]})
        tc.add_task_constraint({'initial_constraints':[{'expression':u, 'reference':[0.0, 0.0]}]})

        tc.add_task_constraint({'final_constraints':[{'expression':x, 'reference':[0.05, 0.1], 'hard':False, 'gain':10}]})
        tc.add_task_constraint({'final_constraints':[{'expression':u, 'reference':[1.0, -1.0], 'hard':False, 'gain':10}]})
        tc.set_ocp_solver('ipopt', solver_options)
        tc.set_discretization_settings({'discretization method': 'multiple shooting', 'horizon size': 2, 'order':1, 'integration':'rk'})
        sol = tc.solve_ocp()
        _, x_val= sol.sample(x, grid='control')
        _, u_val= sol.sample(u, grid='control')
        del tc

        tc = tp.task_context(2)
        x = tc.create_expression('x', 'state', (2, 1))
        u = tc.create_expression('u', 'control', (2, 1))
        tc.set_dynamics(x, u)
        tc.add_regularization(x, 10)
        tc.add_regularization(u, 10)
        tc.add_task_constraint({'initial_constraints':[{'expression':x, 'reference':[0.0, 0.0]}]})
        tc.add_task_constraint({'initial_constraints':[{'expression':u, 'reference':[0.0, 0.0]}]})

        ocp = tc.ocp
        ocp.add_objective(ocp.at_tf(cs.sumsqr(x - [0.05,0.1]))*10)
        ocp.add_objective(ocp.at_tf(cs.sumsqr(u - [1.0,-1.0]))*10)
        tc.set_ocp_solver('ipopt', solver_options)
        tc.set_discretization_settings({'discretization method': 'multiple shooting', 'horizon size': 2, 'order':1, 'integration':'rk'})
        sol = tc.solve_ocp()
        _, x_val2= sol.sample(x, grid='control')
        _, u_val2= sol.sample(u, grid='control')
        self.assertAlmostEqual(x_val[-1,0], x_val2[-1,0], 6, "terminal quadratic constraint on vector state not respected" )
        self.assertAlmostEqual(x_val[-1,1], x_val2[-1,1], 6, "terminal quadratic constraint on vector state not respected" )
        self.assertAlmostEqual(u_val[-1,0], u_val2[-1,0], 6, "final constraint on vector control not respected" )
        self.assertAlmostEqual(u_val[-1,1], u_val[-1,1], 6, "final constraint on vector control not respected" )

        #testing L1 terminal costs
        del tc
        tc = tp.task_context(2)
        x = tc.create_expression('x', 'state', (2, 1))
        u = tc.create_expression('u', 'control', (2, 1))
        tc.set_dynamics(x, u)
        tc.add_regularization(x, 10)
        tc.add_regularization(u, 10)
        tc.add_task_constraint({'initial_constraints':[{'expression':x, 'reference':[0.0, 0.0]}]})
        tc.add_task_constraint({'initial_constraints':[{'expression':u, 'reference':[0.0, 0.0]}]})

        tc.add_task_constraint({'final_constraints':[{'expression':x, 'reference':[0.05, 0.1], 'norm':'L1', 'hard':False, 'gain':1}]})
        tc.add_task_constraint({'final_constraints':[{'expression':u, 'reference':[1.0, -1.0], 'hard':False, 'gain':1}]})
        tc.set_ocp_solver('ipopt', solver_options)
        tc.set_discretization_settings({'discretization method': 'multiple shooting', 'horizon size': 2, 'order':1, 'integration':'rk'})
        sol = tc.solve_ocp()
        _, x_val= sol.sample(x, grid='control')
        _, u_val= sol.sample(u, grid='control')

        tc2 = tp.task_context(2)
        x2 = tc2.create_expression('x', 'state', (2, 1))
        u2 = tc2.create_expression('u', 'control', (2, 1))
        tc2.set_dynamics(x2, u2)
        tc2.add_regularization(x2, 10)
        tc2.add_regularization(u2, 10)
        tc2.add_task_constraint({'initial_constraints':[{'expression':x2, 'reference':[0.0, 0.0]}]})
        tc2.add_task_constraint({'initial_constraints':[{'expression':u2, 'reference':[0.0, 0.0]}]})

        ocp = tc2.ocp
        slack = ocp.variable(2,1)
        ocp.subject_to(-slack <= (ocp.at_tf(x2) - [0.05, 0.1] <= slack))
        ocp.add_objective((slack[0] + slack[1])*1)
        ocp.add_objective(ocp.at_tf(cs.sumsqr(u2 - [1.0,-1.0]))*1)
        tc2.set_ocp_solver('ipopt', solver_options)
        tc2.set_discretization_settings({'discretization method': 'multiple shooting', 'horizon size': 2, 'order':1, 'integration':'rk'})
        sol2 = tc2.solve_ocp()
        _, x_val2= sol2.sample(x2, grid='control')
        _, u_val2= sol2.sample(u2, grid='control')
        self.assertAlmostEqual(x_val[-1,0], x_val2[-1,0], 6, "terminal quadratic constraint on vector state not respected" )
        self.assertAlmostEqual(x_val[-1,1], x_val2[-1,1], 6, "terminal quadratic constraint on vector state not respected" )
        self.assertAlmostEqual(u_val[-1,0], u_val2[-1,0], 6, "final constraint on vector control not respected" )
        self.assertAlmostEqual(u_val[-1,1], u_val[-1,1], 6, "final constraint on vector control not respected" )

    def test_path_constraints(self):
        #Testing the addition of path constraints. compare it with opti that uses multiple-shooting
        print("Not implemented")
if __name__ == '__main__':
    unittest.main()
