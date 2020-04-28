"""Problem module to define specific problems involving tasks."""

from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp, FreeTime
import numpy as np

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime, SingleShooting
import casadi as cs

from collections import namedtuple

_OCPvars = namedtuple("OCPvars", ['q', 'q_dot', 'q_ddot', 'q0', 'q_dot0'])

class TaskContext:
	""" Class for Task context
	The class stores all expressions and constraints relevant to an OCP
	"""

	def __init__(self, time, horizon = 10):
		""" Class constructor - initializes and sets the field variables of the class

		:param time: The length of the time horizon of the OCP.

		"""
		ocp = Ocp(T = time)
		self.ocp = ocp
		self.states = {}
		self.controls = {}
		self.variables = {}
		self.parameters = {}
		self.constraints = {}
		self.monitors = {}
		self.opti = ocp.opti

		self.robots = {}
		self.OCPvars = None
		self.horizon = horizon

	def create_expression(self, name, type, shape):

		""" Creates a symbolic expression for variables in OCP.

		:param name: name of the symbolic variable
		:type name: string

		:param type: type of the symbolic variable. \n
			'state' - a variable that stands for a set of states that evolve over time as the states comprising the dynamical system of the OCP. \n
			'control' - For representing the control actions of the dynamical system of the OCP. \n
			'parameter' - Parameters of the dynamical system. Useful for representing quantities that might change over MPC iterations. eg: the initial conditions of the OCP. \n
			'variable' - A decision variable of the OCP that is not a state or the control action of the dynamical system.
		:type type: string

		:param shape: 2-dimensional tuple that denotes the dimensions of the expression.
		:type shape: tuple of int.

		"""

		ocp = self.ocp

		if type == 'state':
			state = ocp.state(shape[0], shape[1])
			self.states[name] = state

			return state

		elif type == 'control':
			control = ocp.control(shape[0], shape[1])
			self.controls[name] = control

			return control

		elif type == 'parameter':
			parameter = ocp.parameter(shape[0], shape[1])
			self.parameters[name] = parameter

			return parameter

		elif type == 'variable':
			variable = ocp.variable(shape[0], shape[1])
			self.variables[name] = variable

			return variable

		else:

			print("ERROR: expression type undefined")




	def set_dynamics(self, state, state_der):

		""" Set dynamics of state variables of the OCP.

		:param state: expression of the state.
		:type state: state expression

		:param state_der: The derivative of state expression.
		:type state_der: const or another expression variable.

		"""


		ocp = self.ocp
		ocp.set_der(state, state_der)

	def acceleration_resolved(self):

		print("Not yet implemented")

	def torque_resolved(self):

		print("Not implemented")

	def add_regularization(self, expression, gain):

		print("Not yet implemented")

	## Turn on collision avoidance for robot links
	def collision_avoidance_hyperplanes(self, toggle):
		#USE def eval_at_control(self, stage, expr, k): where k is the step of multiple shooting
		# can access using ocp._method.eval...
		# then add this as constraint to opti
		print("Not implemented")



	def add_task_constraint(self, task_spec):

		ocp = self.ocp

		if 'initial_constraints' in task_spec:
			for init_con in task_spec['initial_constraints']:
				#Made an assumption that the initial constraint is always hard
				ocp.subject_to(ocp.at_t0(init_con['expression']) == init_con['reference'])

		if 'final_constraints' in task_spec:
			for final_con in task_spec['final_constraints']:

				if final_con['hard']:
					if 'type' in final_con:
						#When the expression is SE(3) expressed as a 4X4 homogeneous transformation matrix
						if 'Frame' in final_con['type']:
							expression = final_con['expression']
							reference = final_con['reference']
							#hard constraint on the translational componenet
							ocp.subject_to(ocp.at_tf(expression[0:3, 3]) == reference[0:3, 3])
							#hard constraint on the rotational component
							rot_error = cs.mtimes(expression[0:3, 0:3], reference[0:3, 0:3])
							ocp.subject_to(ocp.at_tf(cs.vertcat(rot_error[0,0], rot_error[1,1], rot_error[2,2])) == 1)
					else:
						ocp.subject_to(ocp.at_tf(final_con['expression']) == final_con['reference'])

				else:
					if 'norm' not in final_con or final_con['norm'] == 'L2':
						ocp.add_objective(cs.sumsqr(final_con['expression'] - final_con['reference'])*final_con['gain'])

		if not 'path_constraints' in task_spec:
			return
		for path_con in task_spec['path_constraints']:

			if not 'inequality' in path_con and not 'lub' in path_con:
				if not path_con['hard']:
					if 'norm' not in path_con or path_con['norm'] == 'L2':
						# print('L2 norm added')
						ocp.add_objective(ocp.integral(cs.sumsqr(path_con['expression'] - path_con['reference']))*path_con['gain'])
					elif path_con['norm'] == 'L1':
						# print("L1 norm added")
						ocp.add_objective(ocp.integral( cs.fabs(path_con['reference'] - path_con['expression']))*path_con['gain'])
				elif path_con['hard']:

					ocp.subject_to(path_con['expression'] == path_con['reference'])

			elif 'inequality' in path_con:

				if path_con['hard']:
					ocp.subject_to(path_con['expression'] <= path_con['upper_limits'])
				else:
					con_violation = cs.f_max(path_con['expression'] - path_con['upper_limits'], 0)
					if 'norm' not in path_con or path_con['norm'] == 'L2':
						ocp.add_objective(ocp.integral(con_violation**2)*path_con['gain'])
					elif path_con['norm'] == 'L1':
						ocp.add_objective(ocp.integral(con_violation)*path_con['gain'])

			elif 'lub' in path_con:

				if path_con['hard']:
					ocp.subject_to((path_con['lower_limits'] <= path_con['expression']) <= path_con['upper_limits'])

				else:
					con_violation = cs.f_max(path_con['expression'] - path_con['upper_limits'], 0)
					con_violation = con_violation + cs.f_max(path_con['lower_limits'] - path_con['expression'], 0)
					if 'norm' not in path_con or path_con['norm'] == 'L2':
						ocp.add_objective(ocp.integral(con_violation)*path_con['gain'])

			else:
				print('ERROR: unknown type of path constraint added')

	def generate_function(self, name="opti", save=True, codegen=True):
		opti = self.opti
		func = opti.to_function(name, [opti.p, opti.x, opti.lam_g], [opti.x, opti.lam_g, opti.f]);

		if save == True:
			func.save(name+'.casadi');
		if codegen == True:
			func.generate(name+'.c',{"with_header": True});

	def set_ocp_solver(self, solver, options={}):

		""" Choose the numerical solver for solving the OCP and set the options.

		:param solver: name of the solver. 'ipopt', 'sqpmethod'.
		:type solver: string

		:param options: Dictionary of options for the solver
		:type options: dictionary

		"""

		ocp = self.ocp
		ocp.solver(solver, options)

	def set_discretization_settings(self, settings):

		""" Set the discretization method of the OCP

		:param settings: A dictionary for setting the discretization method of the OCP with the fields and options given below. \n
			'horizon_size' - (int)The number of samples in the OCP. \n
			'discretization method'(string)- 'multiple_shooting' or 'single_shooting'. \n
			'order' (integer)- The order of integration. Minumum one. \n
			'integration' (string)- The numerical integration algorithm. 'rk' - Runge-Kutta4 method.
		:type settings: dictionary

		"""

		ocp = self.ocp
		disc_method = settings['discretization method']
		N = settings['horizon size']

		if 'order' not in settings:
			M = 1
		else:
			M = settings['order']

		if disc_method == 'multiple shooting':
			ocp.method(MultipleShooting(N = N, M = M, intg = settings['integration']))
		elif disc_method == 'single shooting':
			ocp.method(SingleShooting(N = N, M = M, intg = settings['integration']))
		else:
			print("ERROR: discretization with " + settings['discretization_method'] + " is not defined")

	def solve_ocp(self):

		""" solves the ocp and returns the rockit solution object
		"""

		ocp = self.ocp
		sol = ocp.solve()
		return sol


	def add_monitors(self, task_mon):
		print("Not implemented")

	def add_robot(self, robot):
	    self.robots[robot.name] = robot
	    # robot.transcribe(self)

	    # self.sim_system_dyn = robot.sim_system_dyn(self.task_context)

	def set_input_resolution(self, robot):

		if robot.input_resolution == "velocity":

			print("ERROR: Not implemented and probably not recommended")

		elif robot.input_resolution == "acceleration":

			q = self.create_expression('q', 'state', (robot.ndof, 1)) #joint positions over the trajectory
			q_dot = self.create_expression('q_dot', 'state', (robot.ndof, 1)) #joint velocities
			q_ddot = self.create_expression('q_ddot', 'control', (robot.ndof, 1))

			#expressions for initial joint position and joint velocity
			q0 = self.create_expression('q0', 'parameter', (robot.ndof, 1))
			q_dot0 = self.create_expression('q_dot0', 'parameter', (robot.ndof, 1))

			self.set_dynamics(q, q_dot)
			self.set_dynamics(q_dot, q_ddot)

			#add joint position, velocity and acceleration limits
			pos_limits = {'lub':True, 'hard': True, 'expression':q, 'upper_limits':robot.joint_ub, 'lower_limits':robot.joint_lb}
			vel_limits = {'lub':True, 'hard': True, 'expression':q_dot, 'upper_limits':robot.joint_vel_ub, 'lower_limits':robot.joint_vel_lb}
			acc_limits = {'lub':True, 'hard': True, 'expression':q_ddot, 'upper_limits':robot.joint_acc_ub, 'lower_limits':robot.joint_acc_lb}
			joint_constraints = {'path_constraints':[pos_limits, vel_limits, acc_limits]}
			self.add_task_constraint(joint_constraints)

			#adding the initial constraints on joint position and velocity
			joint_init_con = {'expression':q, 'reference':q0}
			joint_vel_init_con = {'expression':q_dot, 'reference':q_dot0}
			init_constraints = {'initial_constraints':[joint_init_con, joint_vel_init_con]}
			self.add_task_constraint(init_constraints)

			self.OCPvars = _OCPvars(q, q_dot, q_ddot, q0, q_dot0)

		elif input_resolution == "torque":

		    print("ERROR: Not implemented")

		else:

		    print("ERROR: Only available options for input_resolution are: \"velocity\", \"acceleration\" or \"torque\".")

		# return _robot.set_input_resolution(task_context = self, input_resolution = input_resolution)

class Point2Point(TaskContext):
	"""Docstring for class Point2Point.

	This class defines a point-to-point motion problem
	by setting its constraints, objective and ocp variables.::

	    from tasho import Problem
	    problem = Point2Point()

	"""
	def __init__(self, time, horizon = 10, goal = None):
		# First call __init__ from TaskContext
		super().__init__(time, horizon)

		if goal is not None:
			if ((isinstance(goal, list) and int(len(goal)) == 3) or
				(isinstance(goal, np.ndarray) and goal.shape == (3, 1))):
				print("Goal position")
			elif (isinstance(goal, np.ndarray) and goal.shape == (4, 4)):
				print("Goal transformation matrix")

		self.goal = goal

	def add_robot(self, robot):
		self.robots[robot.name] = robot
		robot.transcribe(self)

		# TODO: Rethink how much should be included in robot.transcribe method, since this should work for multiple robots
		# Maybe using task.transcribe instead of robot.transcribe is an option

		goal = self.goal

		# Based on Jeroen's example
		SOLVER = "ipopt"
		SOLVER_SETTINGS = {
		    "ipopt": {
		        "max_iter": 1000,
		        "hessian_approximation": "limited-memory",
		        "limited_memory_max_history": 5,
		        "tol": 1e-3,
		    }
		}
		DISC_SETTINGS = {
		    "discretization method": "multiple shooting",
		    "horizon size": self.horizon,
		    "order": 1,
		    "integration": "rk",
		}

		# Set the problem up
		q, q_dot, q_ddot, q0, q_dot0 = self.OCPvars

		# Set expression for End-effector's transformation matrix
		for _key in self.robots:
			_robot = self.robots[_key]
		fk_fun = _robot.fk(q)[7]

		# Define pose constraints
		finalposition_con = {"hard": True, "type": "Frame", "expression": fk_fun, "reference": goal}
		finalvelocity_con = {"hard": True, "expression": q_dot, "reference": 0}
		final_constraints = {"final_constraints": [finalposition_con, finalvelocity_con]}

		# Define path constraints
		vel_regularization = {'hard': False, 'expression':q_dot, 'reference':0, 'gain':1}
		acc_regularization = {'hard': False, 'expression':q_ddot, 'reference':0, 'gain':1}
		path_soft_constraints = {'path_constraints':[vel_regularization, acc_regularization]}

		# Add constraints to task
		self.add_task_constraint(final_constraints)
		self.add_task_constraint(path_soft_constraints)

		# Set settings
		self.set_ocp_solver(SOLVER, SOLVER_SETTINGS)
		self.set_discretization_settings(DISC_SETTINGS)
