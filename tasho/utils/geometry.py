# library for several simple geometric operations
import casadi as cs


def inv_T_matrix(T):

    T_inv = cs.horzcat(
        cs.horzcat(T[0:3, 0:3].T, cs.mtimes(-T[0:3, 0:3].T, T[0:3, 3])).T, [0, 0, 0, 1]
    ).T

    return T_inv


def cross_vec2mat(v, format="MX"):

    """Takes in a 3 dimensonal vector and returns a skew symmetric matrix
    cross product matrix"""

    if format == "MX":
        R = cs.MX(3, 3)

    elif format == "SX":
        R = cs.SX(3, 3)

    elif format == "DM":
        R = cs.DM(3, 3)

    R[0, 1] = -v[2]
    R[0, 2] = v[1]
    R[1, 0] = v[2]
    R[1, 2] = -v[0]
    R[2, 0] = -v[1]
    R[2, 1] = v[0]

    return R


def cross_mat2vec(W, format="MX"):

    """ Takes in a skew-symmetric cross product matrix and returns a vector"""

    v = cs.vertcat(W[2, 1],
        W[0, 2],
        W[1, 0])

    return v


def cross_prod(x, y, format="MX"):

    """ Takes in two vectors and returns their cross product"""

    v = cs.vertcat(x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0])

    return v


def rotmat_to_axisangle(R, reg = 1e-2):

    cos_theta = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2
    unnormalized_axis = cs.vertcat(
        (R[2, 1] - R[1, 2]), (R[0, 2] - R[2, 0]), (R[1, 0] - R[0, 1])
    )
    sin_theta = (cs.sqrt(cs.sumsqr(unnormalized_axis)+reg) - reg**0.5) / 2
    # Adding by a small number to avoid singularity

    theta = cs.atan2(sin_theta, cos_theta)
    axis = unnormalized_axis
    axis = unnormalized_axis / cs.sqrt(cs.sumsqr(unnormalized_axis) + reg*1e-16)

    return theta, axis
