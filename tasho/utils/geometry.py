#library for several simple geometric operations
import casadi as cs


def inv_T_matrix(T):

	T_inv = cs.horzcat(cs.horzcat(T[0:3, 0:3].T, cs.mtimes(-T[0:3, 0:3].T, T[0:3,3])).T, [0, 0, 0, 1]).T

	return T_inv