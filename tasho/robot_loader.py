#Function to load casadi robot models

import casadi as cs

def load_fk(robot_name):

	file = '/robots/' + robot_name + '/' + robot_name + '_fk.casadi'
	robot_fk = cs.Function.load(file)

	return load_fk

def load_inverse_dynamics(robot_name):

	file = '/robots/' + robot_name + '/' + robot_name + '_id.casadi'
	robot_id = cs.Function.load(file)

	return robot_id

def load_forward_dynamics(robot_name):

	file = '/robots/' + robot_name + '/' + robot_name + '_fd.casadi'
	robot_fd = cs.Function.load(file)

	return robot_fd