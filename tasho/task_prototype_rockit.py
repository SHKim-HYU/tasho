#Takes the task-specification and also the task context as an input and 
#returns a COP

from rockit import Ocp, DirectMethod, MultipleShooting, FreeTime
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
			ocp = self.ocp
			control = ocp.control(shape[0], shape[1])
			self.controls[name] = control

			return control

		elif type == 'parameter':

			ocp = self.ocp
			parameter = ocp.parameter(shape[0], shape[1])
			self.parameters[name] = parameter

			return parameter

		else:

			print("ERROR: expression type undefined")

	
	def set_dynamics(self, state, state_der):
	
		ocp = self.ocp
		ocp.set_der(state, state_der)



	def add_task_constraint(self, task_spec):

		ocp = self.ocp
	
		for init_con in task_spec['initial_constraints']:
			#Made an assumption that the initial constraint is always hard
			ocp.subject_to(ocp.at_t0(init_con['expression'], init_con['reference']))
	
		
		for final_con in task_spec['final_constraints']:

			if final_con['hard']:
				ocp.subject_to(ocp.at_tf(final_con['expression'], final_con['reference']))

			else:
				if 'norm' not in final_con or final_con['norm'] == 'L2':
					ocp.add_objective(cs.sumsqr(final_con['expression'] - final_con['reference'])*final_con['gain'])


		for path_con in task_spec['path_constraints']:

			if not path_con['hard']:
				if 'norm' not in path_con or path_con['norm'] == 'L2':
					ocp.add_objective(cs.sumsqr(path_con['expression'] - path_con['reference'])*path_con['gain'])

ocp = Ocp(T = 5)
param = ocp.parameter(5, 5)
print(param.size())