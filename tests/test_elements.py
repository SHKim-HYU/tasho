import unittest
# from tasho import task_prototype_rockit as tp
import casadi as cs
from tasho.TaskModel import Task
from tasho.Variable import Variable
from tasho.Expression import Expression
from tasho.ConstraintExpression import ConstraintExpression
# from tasho.OCPGenerator import OCPGenerator
import numpy as np

class TestElements(unittest.TestCase):
    def test_variables(self):

        x0 = Variable("x0", "Test", "state", (2,1))

        self.assertTrue(x0.mid == 'Test')
        self.assertTrue(x0.name == 'x0')
        self.assertTrue(x0.uid == 'Test_x0')
        self.assertTrue(x0.type == 'state')
        self.assertTrue(x0.shape == (2,1))
        self.assertTrue(type(x0.x) == cs.MX)

        x0.type = 'control'
        self.assertTrue(x0.type == 'control')

        # Create variable inside a task
        task0 = Task("task0", "Test")

        # create a double integrator system
        x = task0.create_variable("x", "Test", "state", (2,1))
        self.assertTrue("Test_x" in task0.variables)



    def test_expressions(self):

        x0 = Variable("x0", "Test", "state", (2,1))

        expr0 = Expression("expr0", "Test", lambda x : cs.sin(x), x0)

        self.assertTrue(expr0.mid == 'Test')
        self.assertTrue(expr0.name == 'expr0')
        self.assertTrue(expr0.uid == 'Test_expr0')
        self.assertTrue(expr0.parents[0].uid == 'Test_x0')
        self.assertTrue(expr0.parent_uid[0] == 'Test_x0')

        task0 = Task("task0", "Test")
        x = task0.create_variable("x", "Test", "state", (2,1))
        
        expr_x_fun = lambda x : cs.vertcat(cs.sin(x[0]), cs.sin(x[1]))

        expr_x = task0.create_expression("x_dot", "Test", expr_x_fun, x)
        self.assertTrue("Test_x_dot" in task0.expressions)

        f0 = cs.Function('f0', [task0.variables['Test_x'].x], [expr_x.evaluate_expression(task0)])

        x_dummy = cs.MX.sym('x_dummy',2,1)
        f1 = cs.Function('f1',[x_dummy], [cs.vertcat(cs.sin(x_dummy[0]), cs.sin(x_dummy[1]))])

        arg_dummy = np.random.rand(2,1)
        
        self.assertTrue((f0(arg_dummy).full() == f1(arg_dummy).full()).all())


    def test_constraint_expressions(self):

        x0 = Variable("x0", "Test", "state", (2,1))

        expr0 = Expression("expr0", "Test", lambda x : cs.sin(x), x0)

        # Test hard box constraint
        con_expr0 = ConstraintExpression("con_expr0", "Test", expr0, 'hard', lb = -0.5, ub = 0.5)

        self.assertTrue(con_expr0.mid == 'Test')
        self.assertTrue(con_expr0.uid == 'Test_con_expr0')
        self.assertTrue(con_expr0.expr == expr0.uid)
        self.assertTrue(con_expr0.constraint_dict['lub'])
        self.assertTrue(con_expr0.constraint_dict['hard'])
        self.assertTrue(con_expr0.constraint_dict['lower_limits'] == -0.5)
        self.assertTrue(con_expr0.constraint_dict['upper_limits'] == 0.5)

        # Test soft equality constraint
        con_expr1 = ConstraintExpression("con_expr1", "Test", expr0, 'soft', reference = -0.5, weight = 1.5, penalty = 'quad')

        self.assertTrue(con_expr1.mid == 'Test')
        self.assertTrue(con_expr1.uid == 'Test_con_expr1')
        self.assertTrue(con_expr1.expr == expr0.uid)
        self.assertTrue(con_expr1.constraint_dict['equality'])
        self.assertTrue(con_expr1.constraint_dict['reference'] == -0.5)
        self.assertFalse(con_expr1.constraint_dict['hard'])
        self.assertTrue(con_expr1.constraint_dict['gain'] == 1.5)
        self.assertTrue(con_expr1.constraint_dict['norm'] == 'quad')

        con_expr1.change_weight(3.5)
        self.assertTrue(con_expr1.constraint_dict['gain'] == 3.5)


        task0 = Task("task0", "Test")
        x = task0.create_variable("x", "Test", "state", (2,1))
        
        expr_x_fun = lambda x : cs.vertcat(cs.sin(x[0]), cs.sin(x[1]))

        expr_x = task0.create_expression("x_dot", "Test", expr_x_fun, x)

        x_con0 = task0.create_constraint_expression("x0_con", "vec_equality", expr_x, 'hard', reference = [0, 0])
        self.assertTrue("vec_equality_x0_con" in task0.constraint_expressions)
        task0.add_initial_constraints(x_con0)
        self.assertTrue(task0.constraints[('initial', 'vec_equality_x0_con')] == ('initial', 'vec_equality_x0_con'))



    def test_constraint_instances(self):

          #creating a task
        task0 = Task("first", "hello_world")

        # create a double integrator system
        x = task0.create_variable("x", "Hello_world", "state", (2,1))
        u = task0.create_variable("u", "Hello_world", "control", (1,1))
        
        dyn_x_fun = lambda x, u : cs.vertcat(x[1], u)
        dyn_x = task0.create_expression("x_dot", "Hello_world", dyn_x_fun, x, u)
        eval_dyn_x = dyn_x.evaluate_expression(task0)
        
        task0.set_der(x, dyn_x)
        
       
        x_con0 = task0.create_constraint_expression("x0_con", "vec_equality", x, 'hard', reference = [0, 0])
        self.assertTrue("vec_equality_x0_con" in task0.constraint_expressions)
        task0.add_initial_constraints(x_con0)
        self.assertTrue(task0.constraints[('initial', 'vec_equality_x0_con')] == ('initial', 'vec_equality_x0_con'))

        x0_expr = task0.create_expression("x_pos", "Hello_world", lambda x: x[0], x)
        self.assertTrue("Hello_world_x_pos" in task0.expressions)
        x_conT = task0.create_constraint_expression("xT_con", "equality", x, 'hard', reference = 1)
        self.assertTrue("equality_xT_con" in task0.constraint_expressions)
        task0.add_terminal_constraints(x_conT)
        self.assertTrue(task0.constraints[('terminal', 'equality_xT_con')] == ('terminal', 'equality_xT_con'))

        con_reg = task0.create_constraint_expression("con_reg", "equality", u, 'soft', reference = 0,  weight = 1e-3)
        self.assertTrue("equality_con_reg" in task0.constraint_expressions)
        task0.add_path_constraints(con_reg)
        self.assertTrue(task0.constraints[('path', 'equality_con_reg')] == ('path', 'equality_con_reg'))


if __name__ == "__main__":
    unittest.main()
