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
		self.variables = {}
		self.parameters = {}
		self.constraints = {}
		self.opti = ocp.opti

	## create_state
	# creates a state variable whose dynamics is known
	def create_state(self, name, shape):
	
		a = 1
	
	def create_parameter(self, name, shape):

		a = 1
	
	def set_dynamics(self, state):
	
		a = 1


	def task_protype(self, task_spec):

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

ocp = Ocp(T = 5)
print(type(ocp.opti.f))