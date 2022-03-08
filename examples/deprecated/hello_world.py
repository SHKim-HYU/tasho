import sys
from tasho import task_prototype_rockit as tp
from rockit import MultipleShooting, Ocp
import matplotlib.pyplot as plt

# Constructs a simple OCP to verify that the basic functions of task_prototype_rockit is in order

if __name__ == "__main__":

    tc = tp.task_context(time = 10, horizon_steps = 5)

    x = tc.create_state("x", (1, 1))

    u = tc.create_control("u", (1, 1))

    tc.set_dynamics(x, u)

    task_spec = {"initial_constraints": [{"expression": x, "reference": 0}]}
    task_spec["path_constraints"] = [
        {"expression": u, "reference": 0, "hard": False, "gain": 1}
    ]
    task_spec["final_constraints"] = [{"expression": x, "reference": 1, "hard": True}]

    tc.add_task_constraint(task_spec)

    # tc.set_ocp_solver("ipopt") #ipopt is set as the default solver
    # disc_settings = {
    #     "discretization method": "single shooting",
    #     "horizon size": 5,
    #     "order": 1,
    #     "integration": "rk",
    # }
    disc_settings = {}
    tc.set_discretization_settings(disc_settings)

    sol = tc.solve_ocp()
    t, x_val = tc.sol_sample(x, grid="control")

    plt.figure()
    plt.plot(t, x_val)
    plt.show(block=True)
