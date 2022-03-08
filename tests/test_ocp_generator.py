import unittest
from tasho import task_prototype_rockit as tp
import casadi as cs
from tasho.TaskModel import Task
from tasho.Variable import Variable
from tasho.OCPGenerator import OCPGenerator
from tasho.ConstraintExpression import ConstraintExpression
from examples.templates.Regularization import Regularization
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


        OCP_gen = OCPGenerator(hello_task, False, {"time_period": 1, "horizon_steps":10})
        self.assertTrue(OCP_gen.tc.horizon[0] == 10)
        self.assertTrue(OCP_gen.tc.ocp_rate == 1/OCP_gen.tc.horizon[0])
        OCP_gen.tc.solve_ocp()

        self.assertTrue(OCP_gen.tc.monitors_configured)

        _, y_sol = OCP_gen.tc.sol_sample(OCP_gen.stage_tasks[0].variables[y.uid].x)
        self.assertTrue((y_sol[-1] == [1, 1]).all())

    def test_append_task(self):
        # Defining a toy task, for a ball that is falling freely in space
        def ball_task(name):
            bouncing_ball = Task(name, "2d_freefall")

            x = bouncing_ball.create_variable('x', 'pos', 'state', (1,1))
            y = bouncing_ball.create_variable('y', 'pos', 'state', (1,1))

            xd = bouncing_ball.create_variable('x', 'vel', 'variable', (1,1))
            yd = bouncing_ball.create_variable('y', 'vel', 'state', (1,1))
            grav = bouncing_ball.create_variable('grav', 'constant', 'magic_number', (1,1), -9.81)

            bouncing_ball.set_der(x, xd)
            bouncing_ball.set_der(y, yd)
            bouncing_ball.set_der(yd, grav)

            y_above_ground = ConstraintExpression("ball_above_ground", "", y, 'hard', lb = 0)
            y_hit_ground = ConstraintExpression("ball_hit_ground", "eq", y, 'hard', reference = 0)

            # adding regularization for numerical reasons, will not affect the dynamics
            bouncing_ball.add_path_constraints(y_above_ground, 
                                            Regularization(x, 1e-3), 
                                            Regularization(y, 1e-3),
                                            Regularization(yd, 1e-3))
            bouncing_ball.add_terminal_constraints(y_hit_ground)

            return bouncing_ball

        horizon_steps = 20
        coeff = 0.9 #coefficient of restitution

        number_bounces = 5
        start_pos = [0,1]
        goal_x = 10

        bounce_tasks = []
        for i in range(number_bounces):
            bounce_tasks.append(ball_task("bounce_" + str(i)))
            

        # add initial pose constraints
        bounce_tasks[0].add_initial_constraints(
            ConstraintExpression("x", "start_pos", bounce_tasks[0].variables['pos_x'], 'hard', reference = start_pos[0]),
            ConstraintExpression("y", "start_pos", bounce_tasks[0].variables['pos_y'], 'hard', reference = start_pos[1]))

        # add position constraint at the end of the bounce`
        bounce_tasks[-1].add_terminal_constraints(
            ConstraintExpression("x", "final_pos_con", bounce_tasks[-1].variables['pos_x'], 'hard', reference = goal_x))

        transcription_options = {"horizon_steps" : horizon_steps}
        OCP_gen = OCPGenerator(bounce_tasks[0], True, transcription_options)

        # append the next bounces to create multi-stage problem
        for i in range(1, number_bounces):
            collision_impulse = [lambda a, b: coeff*a+b, [bounce_tasks[i-1].variables['vel_y']], [bounce_tasks[i].variables['vel_y']], 0]
            x_vel_const = [lambda a, b: a - b, [bounce_tasks[i-1].variables['vel_x']], [bounce_tasks[i-1].variables['vel_x']], 0]
            OCP_gen.append_task(bounce_tasks[i], True, transcription_options, exclude_continuity=[bounce_tasks[i].variables["vel_y"]], generic_inter_stage_constraints= [collision_impulse, x_vel_const])

        tc = OCP_gen.tc
        OCP_gen.tc.solve_ocp()

        self.assertEqual(len(OCP_gen.stages),5)

if __name__ == "__main__":
    unittest.main()
