import unittest
from tasho import task_prototype_rockit as tp


class TestTask(unittest.TestCase):
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

        tc.ocp.set_value(p, 5)
        tc.ocp.set_value(x0, 0.05)
        sol = tc.solve_ocp()
        t, x_val = sol.sample(x, grid="control")

        self.assertAlmostEqual(
            x_val[-1], 2.236067977499, 10, "Final position test failed"
        )
        self.assertEqual(t[-1], 5, "Final time test failed")

        ## Test parameter change
        tc.ocp.set_value(p, 0.9)
        sol = tc.solve_ocp()
        t, x_val = sol.sample(x, grid="control")
        self.assertAlmostEqual(
            x_val[-1], 0.9486832980505, 10, "Final position test failed"
        )
        # self.assertEqual( t[-1], 5, "Final time test failed")


if __name__ == "__main__":
    unittest.main()
