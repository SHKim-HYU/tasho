"""
New hello world examples that use the Abstract tasks.
"""

import casadi as cs
from tasho.TaskModel import Task
from tasho.Variable import Variable
from tasho.OCPGenerator import OCPGenerator
import networkx as nx
# from tasho.Expression import Expression
# from tasho.ConstraintExpression import ConstraintExpression

if __name__ == "__main__":
    
    #creating a task
    hello_task = Task("first", "hello_world")

    # create a double integrator system
    x = hello_task.create_variable("x", "Hello_world", "state", (2,1))
    assert "Hello_world_x" in hello_task.variables
    u = hello_task.create_variable("u", "Hello_world", "control", (1,1))
    assert "Hello_world_u" in hello_task.variables

    dyn_x_fun = lambda x, u : cs.vertcat(x[1], u)
    dyn_x = hello_task.create_expression("x_dot", "Hello_world", dyn_x_fun, x, u)
    dyn_x.evaluate_expression(hello_task)
    hello_task.set_der(x, dyn_x)

    # change the variable x and verify that the expressions dependant also change
    y = Variable("y", "Hello_task", "state", (2,1))
    hello_task.substitute_variable(x, y)
    assert hello_task.variables["Hello_world_x"] == y
    assert (dyn_x._x[0] == x._x[1]) == cs.MX(1)
    dyn_x.evaluate_expression(hello_task)
    assert (dyn_x._x[0] == y._x[1]) == cs.MX(1)
    y = x

    x_con0 = hello_task.create_constraint_expression("x0_con", "vec-equality", y, 'hard', reference = [0, 0])
    hello_task.add_initial_constraints(x_con0)

    x0_expr = hello_task.create_expression("x_pos", "Hello_world", lambda y: y[0], y)
    x_conT = hello_task.create_constraint_expression("xT_con", "equality", y, 'hard', reference = 1)
    hello_task.add_terminal_constraints(x_conT)

    con_reg = hello_task.create_constraint_expression("con_reg", "equality", u, 'soft', reference = 0,  weight = 1e-3)
    hello_task.add_path_constraints(con_reg)

    import matplotlib.pyplot as plt
    plt.plot()
    nx.draw(hello_task.graph, with_labels = True, font_weight = 'bold', node_size = 1000)
    plt.show(block = True)
    #creating an OCP object

    OCP_gen = OCPGenerator(hello_task, False, {"time_period": 1, "horizon_steps":10})
    OCP_gen.tc.solve_ocp()

    _, y_sol = OCP_gen.tc.sol_sample(OCP_gen.stage_tasks[0].variables[y.uid].x)
    print(y_sol)