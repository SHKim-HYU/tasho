import unittest
import casadi as cs
from tasho.utils.symlin import symlin
from random import uniform
import numpy as np

class TestElements(unittest.TestCase):

    def test_symlin(self):

        w = cs.MX.sym('w')
        expr = cs.sin(w)

        expr_p = symlin(expr)

        # Test steps of lin operator

        J_p = cs.Function("J_p",[w],[cs.jacobian(expr_p,w)])
        H_p = cs.Function("H_p",[w],[cs.hessian(expr_p,w)[0]])

        w0 = round(uniform(0.1, 100.0), 2)
        
        self.assertAlmostEqual(J_p(w0),cs.cos(w0),5)

        self.assertAlmostEqual(H_p(w0),0,5)


if __name__ == "__main__":
    unittest.main()
