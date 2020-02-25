#Takes the task-specification and also the task context as an input and
#returns a COP

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime, SingleShooting
import casadi as cs

## Class for task context
# The class stores all expressions and constraints relevant to an OCP
class task_context:

	## Class constructor
	# Initializes and sets the field variables of the class
	def __init__(self, time):
		ocp = Ocp(T = time)
		self.ocp = ocp
		self.states = {}
		self.controls = {}
		self.variables = {}
		self.parameters = {}
		self.constraints = {}
		self.monitors = {}
		self.opti = ocp.opti

	## create_state
	# creates a state variable whose dynamics is known
	def create_expression(self, name, type, shape):

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

		ocp = self.ocp
		ocp.set_der(state, state_der)



	def add_task_constraint(self, task_spec):

		ocp = self.ocp

		if 'initial_constraints' in task_spec:
			for init_con in task_spec['initial_constraints']:
				#Made an assumption that the initial constraint is always hard
				ocp.subject_to(ocp.at_t0(init_con['expression']) == init_con['reference'])

		if 'final_constraints' in task_spec:
			for final_con in task_spec['final_constraints']:

				if final_con['hard']:
					ocp.subject_to(ocp.at_tf(final_con['expression']) == final_con['reference'])

				else:
					if 'norm' not in final_con or final_con['norm'] == 'L2':
						ocp.add_objective(cs.sumsqr(final_con['expression'] - final_con['reference'])*final_con['gain'])


		for path_con in task_spec['path_constraints']:

			if not 'inequality' in path_con and not 'lub' in path_con:
				if not path_con['hard']:
					if 'norm' not in path_con or path_con['norm'] == 'L2':
						ocp.add_objective(ocp.integral(cs.sumsqr(path_con['expression'] - path_con['reference']))*path_con['gain'])

				elif path_con['hard']:

					ocp.subject_to(path_con['expression'] == path_con['reference'])

			elif 'inequality' in path_con:

				if path_con['hard']:
					ocp.subject_to(path_con['expression'] <= path_con['upper_limits'])
				else:
					con_violation = cs.f_max(path_con['expression'] - path_con['upper_limits'], 0)
					if 'norm' not in path_con or path_con['norm'] == 'L2':
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

	def set_ocp_solver(self, solver):

		ocp = self.ocp
		ocp.solver('ipopt')

	def set_discretization_settings(self, settings):

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

		ocp = self.ocp
		sol = ocp.solve()
		return sol


	def add_monitors(self, task_mon):

		print("Not implemented")

if __name__ == '__main__':
	ocp = Ocp(T = 5)
	param = ocp.parameter(5, 5)
	print(param.size())
