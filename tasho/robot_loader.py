#Function to load casadi robot models

import casadi as cs

def load_fk(robot_name):

	file = '/robots/' + robot_name + '/' + robot_name + '_fk.casadi'
	robot_fk = cs.Function.load(file)

	return load_fk
