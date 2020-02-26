# from numpy import sin, cos, tan
from casadi import vertcat, sumsqr, Function
from math import inf
from numbers import Real
import matplotlib.pyplot as plt


class Robot:
    def __init__(self, name="kinova"):
        self.current_X = None
        self.fd = Function.load('./robots/' + name + '/' + name + '_fd.casadi')
        self.id = Function.load('./robots/' + name + '/' + name + '_id.casadi')
        self.fk = Function.load('./robots/' + name + '/' + name + '_fk.casadi')
        self.ndof = self.fk.size1_in(0) # TODO: Improve procedure to determine degrees of freedom
        self.joint_ub = None
        self.joint_lb = None
        self.joint_vel_ub = None
        self.joint_vel_lb = None
        self.joint_acc_ub = None
        self.joint_acc_lb = None
        self.torque_ub = None
        self.torque_lb = None

    def sim_system_dyn(self, ocp):
        # Get discretised dynamics as CasADi function to simulate the system
        sim_system_dyn = ocp._method.discrete_system(ocp)
        return sim_system_dyn

    def set_joint_limits(self, lb = None, ub = None):
        # TODO: This should come from our Pinocchio's interface
        # TODO: Print some warning/error when size of lb and ub doesn't correspond to ndof
        ndof = self.ndof
        if ub == None:
            _ub = inf
            for i in range(1, ndof):
                _ub = vertcat(_ub,inf)
        elif isinstance(ub, Real):
            _ub = ub
        else:
            if len(ub) != ndof:
                _ub = inf
                for i in range(1, ndof):
                    _ub = vertcat(_ub,inf)
            else:
                _ub = ub

        if lb == None:
            _lb = -inf
            for i in range(1, ndof):
                _lb = vertcat(_lb,-inf)
        elif isinstance(lb, Real):
            _lb = lb
        else:
            if len(lb) != ndof:
                _lb = -inf
                for i in range(1, ndof):
                    _lb = vertcat(_lb,-inf)
            else:
                _lb = lb

        self.joint_ub = _ub
        self.joint_lb = _lb

    def set_torque_limits(self, lb = None, ub = None):
        # TODO: This should come from our Pinocchio's interface
        # TODO: Print some warning/error when size of lb and ub doesn't correspond to ndof
        ndof = self.ndof
        if ub == None:
            _ub = inf
            for i in range(1, ndof):
                _ub = vertcat(_ub,inf)
        elif isinstance(ub, Real):
            _ub = ub
        else:
            if len(ub) != ndof:
                _ub = inf
                for i in range(1, ndof):
                    _ub = vertcat(_ub,inf)
            else:
                _ub = ub

        if lb == None:
            _lb = -inf
            for i in range(1, ndof):
                _lb = vertcat(_lb,-inf)
        elif isinstance(lb, Real):
            _lb = lb
        else:
            if len(lb) != ndof:
                _lb = -inf
                for i in range(1, ndof):
                    _lb = vertcat(_lb,-inf)
            else:
                _lb = lb

        self.torque_ub = _ub
        self.torque_lb = _lb

    def set_joint_velocity_limits(self, lb = None, ub = None):
        # TODO: This should come from our Pinocchio's interface
        # TODO: Print some warning/error when size of lb and ub doesn't correspond to ndof
        ndof = self.ndof
        if ub == None:
            _ub = inf
            for i in range(1, ndof):
                _ub = vertcat(_ub,inf)
        elif isinstance(ub, Real):
            _ub = ub
        else:
            if len(ub) != ndof:
                _ub = inf
                for i in range(1, ndof):
                    _ub = vertcat(_ub,inf)
            else:
                _ub = ub

        if lb == None:
            _lb = -inf
            for i in range(1, ndof):
                _lb = vertcat(_lb,-inf)
        elif isinstance(lb, Real):
            _lb = lb
        else:
            if len(lb) != ndof:
                _lb = -inf
                for i in range(1, ndof):
                    _lb = vertcat(_lb,-inf)
            else:
                _lb = lb

        self.joint_vel_ub = _ub
        self.joint_vel_lb = _lb

    def set_joint_acceleration_limits(self, lb = None, ub = None):
        # TODO: This should come from our Pinocchio's interface
        # TODO: Print some warning/error when size of lb and ub doesn't correspond to ndof
        ndof = self.ndof
        if ub == None:
            _ub = inf
            for i in range(1, ndof):
                _ub = vertcat(_ub,inf)
        elif isinstance(ub, Real):
            _ub = ub
        else:
            if len(ub) != ndof:
                _ub = inf
                for i in range(1, ndof):
                    _ub = vertcat(_ub,inf)
            else:
                _ub = ub

        if lb == None:
            _lb = -inf
            for i in range(1, ndof):
                _lb = vertcat(_lb,-inf)
        elif isinstance(lb, Real):
            _lb = lb
        else:
            if len(lb) != ndof:
                _lb = -inf
                for i in range(1, ndof):
                    _lb = vertcat(_lb,-inf)
            else:
                _lb = lb

        self.joint_acc_ub = _ub
        self.joint_acc_lb = _lb

    @property
    def get_initial_conditions(self):
        return self.current_X


    # def sim_system_dyn(self, ocp):
    #     # Get discretised dynamics as CasADi function to simulate the system
    #     sim_system_dyn = ocp._method.discrete_system(ocp)
    #     return sim_system_dyn


# class TwoLinkPlanar(Robot):
#     def __init__(self, r_veh=1., bounds={}, safety_dist=0., look_ahead_distance=5.):
#         self.r_veh = r_veh
#         self.bounds = bounds
#         self.safety_dist = safety_dist
#         self.look_ahead_distance = look_ahead_distance
#         self.current_X = vertcat(0, 0, 0)
#         self.n_states = 4
#         self.n_controls = 2
#
#     def transcribe(self, ocp):
#         bounds = self.bounds
#
#         # Define states
#         q1          = ocp.state()
#         q2          = ocp.state()
#         dq1         = ocp.state()
#         dq2         = ocp.state()
#         self.q1     = q1
#         self.q2     = q2
#         self.dq2    = dq2
#         self.dq2    = dq2
#
#         # Define controls
#         tau1        = ocp.control()
#         tau2        = ocp.control()
#         self.tau1   = tau1
#         self.tau2   = tau2
#
#         print('The two-link planar robot model defines four states (q1, q2, dq1, dq2) and two control inputs (tau1 and tau2)')
#
#         # Specify vehicle model
#         ocp.set_der(x,     V*cos(theta))
#         ocp.set_der(y,     V*sin(theta))
#         ocp.set_der(theta, dtheta)
#
#         # Path constraints
#         ocp.subject_to(bounds["dthetamin"] <= (dtheta <= bounds["dthetamax"]))
#         ocp.subject_to(bounds["vmin"] <= (V <= bounds["vmax"]))
#
#         # Initial guess
#         ocp.set_initial(dtheta, 0)
#         ocp.set_initial(V,      bounds["vmax"])
#
#         # Define parameter for initial state
#         X0 = ocp.parameter(3)
#         self.X0 = X0
#
#         # Initial constraints
#         X = vertcat(x, y, theta)
#         ocp.subject_to(ocp.at_t0(X) == X0)
#
#         # Add penalty on turning
#         ocp.add_objective(1*ocp.sum(sumsqr(dtheta), grid='control'))
#
#     def set_start_pose(self, ocp, states):
#         self.current_X = vertcat(states)
#         ocp.set_value(self.X0, self.current_X)
#
#     def set_initial_guess(self, ocp, states):
#         ocp.set_initial(self.x,     states[0])
#         ocp.set_initial(self.y,     states[1])
#         ocp.set_initial(self.theta, states[2])
#
#     @property
#     def get_states(self):
#         return vertcat(self.x, self.y, self.theta)
#
#     @property
#     def get_controls(self):
#         return vertcat(self.dtheta, self.V)
#
#     def get_output_states(self, ocp):
#         return vertcat(ocp.sample(self.x)[1], ocp.sample(self.y)[1], ocp.sample(self.theta)[1])
#
#     def get_output_controls(self, ocp):
#         return vertcat(ocp.sample(self.dtheta)[1], ocp.sample(self.V)[1])
#Function to load casadi robot models

# import casadi as cs
#
# def load_fk(robot_name):
#
# 	file = '/robots/' + robot_name + '/' + robot_name + '_fk.casadi'
# 	robot_fk = cs.Function.load(file)
#
# 	return load_fk
#
# def load_inverse_dynamics(robot_name):
#
# 	file = '/robots/' + robot_name + '/' + robot_name + '_id.casadi'
# 	robot_id = cs.Function.load(file)
#
# 	return robot_id
#
# def load_forward_dynamics(robot_name):
#
# 	file = '/robots/' + robot_name + '/' + robot_name + '_fd.casadi'
# 	robot_fd = cs.Function.load(file)
#
# 	return robot_fd
