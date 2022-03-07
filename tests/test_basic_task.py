import unittest
from tasho import task_prototype_rockit as tp
import casadi as cs
from tasho.TaskModel import Task
from tasho.Variable import Variable
from tasho.OCPGenerator import OCPGenerator
import numpy as np

class TestTask(unittest.TestCase):
    def test_abstract_task(self):

        #creating a task
        hello_task = Task("first", "hello_world")

        # create a double integrator system
        x = hello_task.create_variable("x", "Hello_world", "state", (2,1))
        self.assertTrue("Hello_world_x" in hello_task.variables)
        u = hello_task.create_variable("u", "Hello_world", "control", (1,1))
        self.assertTrue("Hello_world_u" in hello_task.variables)

        dyn_x_fun = lambda x, u : cs.vertcat(x[1], u)

        dyn_x = hello_task.create_expression("x_dot", "Hello_world", dyn_x_fun, x, u)
        self.assertTrue("Hello_world_x_dot" in hello_task.expressions)

        eval_dyn_x = dyn_x.evaluate_expression(hello_task)
        self.assertEqual(str(eval_dyn_x), 'vertcat(Hello_world_x[1], Hello_world_u)')

        hello_task.set_der(x, dyn_x)
        self.assertTrue("Hello_world_x" in hello_task._state_dynamics)

        # change the variable x and verify that the expressions dependant also change
        y = Variable("y", "Hello_world", "state", (2,1))
        hello_task.substitute_expression(x, y)
        self.assertFalse("Hello_world_x" in hello_task.variables)
        self.assertTrue("Hello_world_y" in hello_task.variables)
        hello_task.set_der(y, dyn_x)
        self.assertTrue("Hello_world_y" in hello_task._state_dynamics)

        x_con0 = hello_task.create_constraint_expression("x0_con", "vec_equality", y, 'hard', reference = [0, 0])
        self.assertTrue("vec_equality_x0_con" in hello_task.constraint_expressions)
        hello_task.add_initial_constraints(x_con0)
        self.assertTrue(hello_task.constraints[('initial', 'vec_equality_x0_con')] == ('initial', 'vec_equality_x0_con'))

        x0_expr = hello_task.create_expression("x_pos", "Hello_world", lambda y: y[0], y)
        self.assertTrue("Hello_world_x_pos" in hello_task.expressions)
        x_conT = hello_task.create_constraint_expression("xT_con", "equality", y, 'hard', reference = 1)
        self.assertTrue("equality_xT_con" in hello_task.constraint_expressions)
        hello_task.add_terminal_constraints(x_conT)
        self.assertTrue(hello_task.constraints[('terminal', 'equality_xT_con')] == ('terminal', 'equality_xT_con'))

        con_reg = hello_task.create_constraint_expression("con_reg", "equality", u, 'soft', reference = 0,  weight = 1e-3)
        self.assertTrue("equality_con_reg" in hello_task.constraint_expressions)
        hello_task.add_path_constraints(con_reg)
        self.assertTrue(hello_task.constraints[('path', 'equality_con_reg')] == ('path', 'equality_con_reg'))



    def test_task_hw(self):
        tc = tp.task_context(5, horizon_steps=5)

        x, x0 = tc.create_state("x", init_parameter=True)
        u = tc.create_control("u")
        p = tc.create_parameter("p")

        tc.set_dynamics(x, u)

        task_spec = {}
        task_spec["path_constraints"] = [
            {"expression": u, "reference": 0, "hard": False, "gain": 1}
        ]
        task_spec["final_constraints"] = [
            {"expression": x ** 2, "reference": p, "hard": True}
        ]

        tc.add_task_constraint(task_spec)

        solver_options = {
            "ipopt": {"print_level": 0},
            "print_time": False,
            "expand": True,
        }

        tc.set_ocp_solver("ipopt", solver_options)
        disc_settings = {
            "discretization method": "multiple shooting",
            "order": 2,
            "integration": "rk",
        }
        tc.set_discretization_settings(disc_settings)

        tc.set_value(p, 5)
        tc.set_value(x0, 0.05)
        sol = tc.solve_ocp()
        t, x_val = tc.sol_sample(x, grid="control")

        self.assertAlmostEqual(
            x_val[-1], 2.236067977499, 10, "Final position test failed"
        )
        self.assertEqual(t[-1], 5, "Final time test failed")

        ## Test parameter change
        tc.set_value(p, 0.9)
        sol = tc.solve_ocp()
        t, x_val = tc.sol_sample(x, grid="control")
        self.assertAlmostEqual(
            x_val[-1], 0.9486832980505, 10, "Final position test failed"
        )
        # self.assertEqual( t[-1], 5, "Final time test failed")


if __name__ == "__main__":
    unittest.main()
