from tasho.ConstraintExpression import ConstraintExpression
from tasho.Expression import Expression
from tasho.utils.geometry import rotmat_to_axisangle2

def ConstraintSE3(expression, reference, rot_tol = 1e-2):

    assert expression.shape == (4,4), "The SE3 expression does not have the 4X4 shape"
    assert reference.shape == (4,4), "The SE3 reference does not have the 4X4 shape"

    # @TODO: assert that both the poses are defined w.r.t to the same inertial reference frame

    error_position = Expression(expression.uid + "_vs_goal", "trans_error", lambda e, r : -e[0:3,3] + r[0:3,3], expression, reference)
    error_rot = Expression(expression.uid + "_vs_goal", "rotmat_error", lambda e, r : e[0:3,0:3].T@r[0:3,0:3], expression, reference)
    error_axang = Expression(expression.uid + "_vs_goal", "ax_ang_error", lambda ep : rotmat_to_axisangle2(ep[0:3, 0:3])[0], error_rot)

    con_trans = ConstraintExpression(expression.uid + "_vs_goal", "trans_con", error_position, "hard", reference = [0,0,0])
    con_rot = ConstraintExpression(expression.uid + "_vs_goal", "rot_con", error_axang, "hard", ub = rot_tol)

    return con_trans, con_rot
