import unittest
from tasho.utils import geometry
import casadi as cs
import numpy as np


class TestTopt(unittest.TestCase):

    np.random.seed(0)

    def test_T_inv(self):

        # Test the inverse operation of the transformation matrix

        for i in range(10):
            R = np.random.random((3,3))
            t = np.random.random((3,1))

            T = cs.horzcat(R, t)

            T_inv = geometry.inv_T_matrix(T)
            # Simply checking the formula
            T_inv_formula = cs.horzcat(R.T, -R.T@t)
            self.assertAlmostEqual(cs.mmax(cs.fabs(T_inv[0:3,:] - T_inv_formula)), 0, 14)

    def test_cross_vec2mat(self):

        # Test the skew operation

        for i in range(10):

            r = np.random.random((3,1))

            R_skew = np.array([[0, -r[2,0], r[1,0]], [r[2,0], 0, -r[0,0]], [-r[1,0], r[0,0], 0]])

            R_skew_geom = geometry.cross_vec2mat(r, "SX")
            self.assertEqual(cs.mmax(cs.fabs(R_skew - cs.DM(R_skew_geom))), 0)

            #also test cross_mat2vec

            r_col = geometry.cross_mat2vec(R_skew_geom)
            self.assertEqual(cs.mmax(cs.fabs(r_col - r)), 0)

    def test_rotmat_to_axisangle(self):

        R = np.eye(3,3)

        theta, axis = geometry.rotmat_to_axisangle(R)
        self.assertAlmostEqual(theta,0, 9)
        self.assertAlmostEqual(cs.mmax(cs.fabs(axis - 0)).full(),0, 9)
        print(theta)
        print(axis)

        R = np.array([[  0.1370971, -0.4458728,  0.8845348],
   [0.6450042,  0.7178971,  0.2619033],
  [-0.7517806,  0.5346225,  0.3860114 ]])

        theta, axis = geometry.rotmat_to_axisangle(R, 0)

        self.assertAlmostEqual(theta, 1.45, 7)
        self.assertAlmostEqual(cs.mmax(cs.fabs(axis.full() - cs.DM([ 0.1373606, 0.8241634, 0.5494423 ]))).full()[0,0],0, 6)

    def test_cross_prod(self):

        # Testing the cross product code

        #cross product with itself gives zero
        v1 = np.random.random((3,1))
        v3 = geometry.cross_prod(v1, v1)
        self.assertEqual(cs.mmax(v3), 0)

        v1 = np.array([[0.92533324],
        [0.69774234],
        [0.39976964]])

        v2 = np.array([[0.7728786 ],
        [0.5005966 ],
        [0.20464854]])

        v3 = np.array([[-0.05733137139004038],[0.1196053031062344],[-0.07605144908894013]])

        vcross = geometry.cross_prod(v1, v2)

        self.assertAlmostEqual(cs.mmax(cs.fabs(v3 - vcross)), 0)



if __name__ == "__main__":
    unittest.main()
