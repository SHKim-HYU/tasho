#File to set the MPC options and to deploy it. Takes as input the task context.
#Provides options for code-generation for deployment on the robot
#Could also optionally create a rosnode that communicates with world_simulator for verifying the MPC in simulations
#Monitor the variables to raise events


class MPC_handler:
    def __init__(self, tc):

# TODO: set a method to let the user define the inputs and outputs of the function get from opti.to_function
# TODO: This should also account for monitors
# TODO: Set offline solution for initialization ( mpc.set_offline_solution(solver, options, ?))
