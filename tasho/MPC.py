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

    def runMPC(self):

    	#First solving OCP for initialization for MPC hotstart


    	#TODO: change by adding termination criteria
    	for mpc_iter in range(10): 
    		print("Not implemented")
    
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





# TODO: set a method to let the user define the inputs and outputs of the function get from opti.to_function
# TODO: This should also account for monitors
# TODO: Set offline solution for initialization ( mpc.set_offline_solution(solver, options, ?))

if __name__ == '__main__':

	print("No syntax errors")
