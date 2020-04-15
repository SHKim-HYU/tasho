#File to set the MPC options and to deploy it. Takes as input the task context.
#Provides options for code-generation for deployment on the robot
#Could also optionally create a rosnode that communicates with world_simulator for verifying the MPC in simulations
#Monitor the variables to raise events


class MPC:
	## MPC(tc, type)
	# @params tc task context defined using the task protype function
	# @params type Specifies the type of MPC interaction. 'bullet_notrealtime' - Here
	# the bullet environment is simulated onyl after the MPC output is computed. So no computation time issues.
	# 'bullet_realtime' - The MPC computation and bullet simulation happens at the same time and they communicate through
	# @params parameters Extra details for the MPC. Such as the mapping between the parameter variables and the data from bullet

    def __init__(self, tc, sim_type, parameters):

    	self.tc = tc
    	self.type = sim_type
    	self.parameters = parameters
    	self.params_names = tc.parmaters.keys()

    	if sim_type == "bullet_notrealtime":

    		self.world = parameters['world'] #object of world_simulator class

    	elif sim_type == "bullet_realtime":

    		print("Not implemented")

    	else:
    		print("[ERROR] Unknown simulation type")

    ## Configures the MPC from the current positions
    def configMPC_fromcurrent(self, init_guess = None):

    	#SOLVE the OCP in order to warm start the MPC

    	tc = self.tc
    	params_val = self._read_params_nrbullet()

    	#set the parameter values
    	for params_name in self.params_name:

    		tc.ocp.set_value(tc.parameters[params_name], params_val[params_name])

    	#set the initial guesses

    	if init_guess != None:

    		print("Not implemented")

    	#For initial guesses, setting a robust solver (IPOPT)
    	tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})

    	#assuming that the discretization settings are already done!

    	sol = tc.solve()
    	sol_states, sol_controls, sol_variables = self._read_solveroutput(sol)
    	self.sol_ocp = [sol_states, sol_controls, sol_variables]

    ## obtain the solution of the ocp
    def _read_solveroutput(self, sol):

    	sol_states = {}
    	sol_controls = {}
    	sol_variables = {}

    	for state in tc.states:
    		_, sol_state = sol.sample(state, grid = 'control')
    		sol_states[state] = sol_state

    	for control in tc.controls:
    		_, sol_control = sol.sample(control, grid = 'control')
    		sol_controls[control] = sol_control

    	for variable in tc.variables:
    		_, sol_variable = sol.sample(variable, grid = 'control')
    		sol_variables[variable] = sol_variable

    	return sol_states, sol_controls, sol_variables

    #Continuous running of the MPC
    def runMPC(self):

    	sol_states = self.sol_ocp[0]
    	sol_controls = self.sol_ocp[1]
    	sol_variables = self.sol_ocp[2]

    	tc = self.tc

    	#TODO: change by adding termination criteria
    	for mpc_iter in range(10):

    		if self.type == "bullet_notrealtime":

    			#reading and setting the latest parameter values
    			params_val = self._read_params_nrbullet()
    			for params_name in self.params_names:
    				tc.set_value(tc.parameters[params_name], params_val[params_name])

    			#set the states, controls and variables as initial values
    			print("Warm starting not implemented!!!!!!!!!!")

    			if self.parameters['solver_name'] == 'ipopt':

    				if 'lbfgs' in self.parameters['solver_params']:

    					tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})

                elif self.parameters['solver_name'] == 'sqpmethod':

                    kkt_tol_pr = 1e-6
                    kkt_tol_du = 1e-6
                    min_step_size = 1e-16
                    max_iter = 1
                    max_iter_ls = 0
                    qpsol_options = {'constr_viol_tol': kkt_tol_pr, 'dual_inf_tol': kkt_tol_du, 'verbose' : False, 'print_iter': False, 'print_header': False, 'dump_in': False} # "error_on_fail" : False
                    solver_options = {'qpsol': 'qrqp', 'qpsol_options': qpsol_options, 'verbose': False, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True}
                    tc.set_ocp_solver('sqpmethod', solver_options)

                    #   {"qpsol": "qrqp","qpsol_options": qpsol_options,"print_header":False,"print_iteration":False,"print_time":False}
                    #   {"qpsol": "qrqp","max_iter_ls":0,"qpsol_options": qpsol_options,"print_header":False,"print_iteration":False,"print_time":False}
                    #   {"qpsol": "qrqp","convexify_strategy":"regularize","max_iter":500,"qpsol_options": qpsol_options,"print_header":False,"print_iteration":True,"print_time":False,"tol_du":1e-8,"min_step_size":1e-12}


                else:
                    # Set ipopt as default solver
                    tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})

    			sol = tc.solve()
    			sol_states, sol_controls, sol_variables = self._read_solveroutput(sol)

    			##TODO: apply the control action to bullet environment


    		elif self.type == "bullet_realtime":

    			print("Not implemented")

    		else:

    			print("[ERROR] Unknown simulation type")

    # Internal function to read the values of the parameter variables from the bullet simulation environment
    # in non realtime case
    def _read_params_nrbullet(self):

    	params_val = {}
    	parameters = self.parameters

    	for params_name in self.params_names:

    		param_info = parameters['params'][params_names]
    		if param_info['type'] == 'joint_position':

    			param_val = []
    			jointsInfo = self.world.readJointState(param_info['robotID'], param_info['joint_indices'])
    			for jointInfo in jointsInfo:
    				param_val.append(jointInfo[0])

    			params_val[params_name] = param_val

    		elif param_info['type'] == 'joint_velocity':

    			param_info = parameters['params'][params_names]
    			jointsInfo = self.world.readJointState(param_info['robotID'], param_info['joint_indices'])
    			for jointInfo in jointsInfo:
    				param_val.append(jointInfo[1])

    			params_val[params_name] = param_val


    		elif param_info['type'] == 'joint_torque':

    			param_info = parameters['params'][params_names]
    			jointsInfo = self.world.readJointState(param_info['robotID'], param_info['joint_indices'])
    			for jointInfo in jointsInfo:
    				param_val.append(jointInfo[3])

    			params_val[params_name] = param_val

    		else:

    			print("[ERROR] Invalid type of parameter to be read from the simulation environment")


    	return params_val


# TODO: set a method to let the user define the inputs and outputs of the function get from opti.to_function
# TODO: This should also account for monitors
# TODO: Set offline solution for initialization ( mpc.set_offline_solution(solver, options, ?))

if __name__ == '__main__':

	print("No syntax errors")
