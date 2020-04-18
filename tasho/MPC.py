#File to set the MPC options and to deploy it. Takes as input the task context.
#Provides options for code-generation for deployment on the robot
#Could also optionally create a rosnode that communicates with world_simulator for verifying the MPC in simulations
#Monitor the variables to raise events

import casadi as cs
import numpy as np


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
        self.params_names = tc.parameters.keys()

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
        self.mpc_ran = False
        params_val = self._read_params_nrbullet()

        #set the parameter values
        for params_name in self.params_names:

            tc.ocp.set_value(tc.parameters[params_name], params_val[params_name])

        #set the initial guesses

        if init_guess != None:

            print("Not implemented")

        #For initial guesses, setting a robust solver (IPOPT)
        tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})

        #assuming that the discretization settings are already done!
        print(tc.ocp._method.M)
        tc.set_discretization_settings(self.parameters['disc_settings'])

        sol = tc.solve_ocp()
        print(tc.ocp._method.N)
        sol_states, sol_controls, sol_variables = self._read_solveroutput(sol)
        self.sol_ocp = [sol_states, sol_controls, sol_variables]

        print(sol_controls['s_ddot'])
        

    ## obtain the solution of the ocp
    def _read_solveroutput(self, sol):

        sol_states = {}
        sol_controls = {}
        sol_variables = {}
        tc = self.tc

        for state in tc.states:
            _, sol_state = sol.sample(tc.states[state], grid = 'control')
            sol_states[state] = sol_state

        for control in tc.controls:
            _, sol_control = sol.sample(tc.controls[control], grid = 'control')
            sol_controls[control] = sol_control

        for variable in tc.variables:
            _, sol_variable = sol.sample(tc.variables[variable], grid = 'control')
            sol_variables[variable] = sol_variable

        return sol_states, sol_controls, sol_variables

    ## Function to provide the initial guess for warm starting the states, controls and variables in tc
    def _warm_start(self, sol_ocp, options = 'reuse'):

        tc = self.tc
        #reusing the solution from previous MPC iteration for warm starting
        if self.mpc_ran == False or  options == 'reuse':

            sol_states = sol_ocp[0]
            for state in tc.states:
                tc.ocp.set_initial(tc.states[state], sol_states[state].T)

            sol_controls = sol_ocp[1]
            for control in tc.controls:
                tc.ocp.set_initial(tc.controls[control], sol_controls[control].T)

            sol_variables = sol_ocp[2]
            for variable in tc.variables:
                tc.ocp.set_initial(tc.variables[variable], sol_variables[variable])

        #warm starting by shiting the solution by 1 step
        elif options == 'shift':

            sol_states = sol_ocp[0]
            for state in tc.states:
                # print(state)
                # print(sol_states[state][1:].shape)
                # print(sol_states[state][-1:].shape)
                tc.ocp.set_initial(tc.states[state], cs.vertcat(sol_states[state][1:], sol_states[state][-1:]).T)
                # tc.ocp.set_initial(tc.states[state], sol_states[state].T)

            sol_controls = sol_ocp[1]
            for control in tc.controls:
                # tc.ocp.set_initial(tc.controls[control], sol_controls[control].T)
                # print(control)
                # print(sol_controls[control][1:-1].shape)
                zeros_np = np.zeros(sol_controls[control][-1:].shape) 
                # print(zeros_np.shape)
                tc.ocp.set_initial(tc.controls[control], cs.vertcat(sol_controls[control][1:-1], zeros_np, zeros_np).T)

            sol_variables = sol_ocp[2]
            for variable in tc.variables:
                tc.ocp.set_initial(tc.variables[variable], sol_variables[variable])

        
        else:

            raise Exception('Invalid MPC restart option ' + options)



    #Continuous running of the MPC
    def runMPC(self):

        sol_states = self.sol_ocp[0]
        sol_controls = self.sol_ocp[1]
        sol_variables = self.sol_ocp[2]

        tc = self.tc

        #TODO: change by adding termination criteria
        for mpc_iter in range(20):

            if self.type == "bullet_notrealtime":

                #reading and setting the latest parameter values
                params_val = self._read_params_nrbullet()
                for params_name in self.params_names:
                    tc.ocp.set_value(tc.parameters[params_name], params_val[params_name])

                #set the states, controls and variables as initial values
                self._warm_start([sol_states, sol_controls, sol_variables], options = 'shift')

                if self.parameters['solver_name'] == 'ipopt':

                    if 'lbfgs' in self.parameters['solver_params']:

                        tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3, 'print_level':0}})

                elif self.parameters['solver_name'] == 'sqpmethod':

                    if 'qrqp' in self.parameters['solver_params']:
                        kkt_tol_pr = 1e-3
                        kkt_tol_du = 1e-1
                        min_step_size = 1e-4
                        max_iter = 1
                        max_iter_ls = 2
                        qpsol_options = {'constr_viol_tol': kkt_tol_pr, 'dual_inf_tol': kkt_tol_du, 'verbose' : False, 'print_iter': False, 'print_header': False, 'dump_in': False, "error_on_fail" : False}
                        solver_options = {'qpsol': 'qrqp', 'qpsol_options': qpsol_options, 'verbose': False, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True} # "convexify_strategy":"regularize"
                        tc.set_ocp_solver('sqpmethod', solver_options)

                    elif 'osqp' in self.parameters['solver_params']:
                        kkt_tol_pr = 1e-3
                        kkt_tol_du = 1e-1
                        min_step_size = 1e-4
                        max_iter = 5
                        max_iter_ls = 2
                        eps_abs = 1e-5
                        eps_rel = 1e-5
                        qpsol_options = {'osqp': {'alpha': 1, 'eps_abs': eps_abs, 'eps_rel': eps_rel, 'verbose':0}, 'dump_in': False, 'error_on_fail':False}
                        solver_options = {'qpsol': 'osqp', 'qpsol_options': qpsol_options, 'verbose': False, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True} # "convexify_strategy":"regularize"
                        tc.set_ocp_solver('sqpmethod', solver_options)

                    elif 'qpoases' in self.parameters['solver_params']:
                        kkt_tol_pr = 1e-3
                        kkt_tol_du = 1e-1
                        min_step_size = 1e-4
                        max_iter = 10
                        max_iter_ls = 0
                        qpoases_tol = 1e-4
                        qpsol_options = {'printLevel': 'none', 'enableEqualities': True, 'initialStatusBounds' : 'inactive', 'terminationTolerance': qpoases_tol}
                        solver_options = {'qpsol': 'qpoases', 'qpsol_options': qpsol_options, 'verbose': False, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True} # "convexify_strategy":"regularize"
                        tc.set_ocp_solver('sqpmethod', solver_options)

                    elif 'ipopt' in self.parameters['solver_params']:
                        kkt_tol_pr = 1e-3
                        kkt_tol_du = 1e-1
                        min_step_size = 1e-6
                        max_iter = 3
                        max_iter_ls = 0

                        ipopt_tol = 1e-3
                        tiny_step_tol = 1e-6
                        mu_init = 1e-3
                        # linear_solver = 'ma27'
                        linear_solver = 'mumps'

                        ipopt_options = {'tol': ipopt_tol, 'tiny_step_tol': tiny_step_tol, 'fixed_variable_treatment': 'make_constraint', 'hessian_constant': 'yes', 'jac_c_constant': 'yes', 'jac_d_constant': 'yes', 'accept_every_trial_step': 'yes', 'mu_init': mu_init, 'print_level': 0, 'linear_solver': linear_solver}
                        nlpsol_options = {'ipopt': ipopt_options, 'print_time': False}
                        qpsol_options = {'nlpsol': 'ipopt', 'nlpsol_options': nlpsol_options, 'print_time': False, 'verbose': False}
                        solver_options = {'qpsol': 'nlpsol', 'qpsol_options': qpsol_options, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True} # "convexify_strategy":"regularize"
                        tc.set_ocp_solver('sqpmethod', solver_options)


                else:
                    # Set ipopt as default solver
                    print("Using IPOPT with LBFGS as the default solver")
                    tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})

                sol = tc.solve_ocp()
                sol_states, sol_controls, sol_variables = self._read_solveroutput(sol)
                sol_mpc = [sol_states, sol_controls, sol_variables]
                self.sol_mpc = sol_mpc
                self.mpc_ran = True
                # Apply the control action to bullet environment
                self._apply_control_nrbullet(sol_mpc)

            elif self.type == "bullet_realtime":

                print("Not implemented")

            else:

                print("[ERROR] Unknown simulation type")

    # Internal function to apply the output of the MPC to the non-realtime bullet environment
    def _apply_control_nrbullet(self, sol_mpc):

        if self.parameters['control_type'] == 'joint_velocity':
            control_info = self.parameters['control_info']

            joint_indices = control_info['joint_indices']
            if control_info['discretization'] =='constant_acceleration':
                #Computing the average of the first two velocities to apply as input
                #assuming constant acceleration input
                control_action = 0.5*(sol_mpc[0]['q_dot'][0] + sol_mpc[0]['q_dot'][1])

            self.world.setController(control_info['robotID'], 'velocity', joint_indices, targetVelocities = control_action)
            print("This ran")
            print(sol_mpc[0]['s_dot'])
            print(sol_mpc[0]['s'])
        elif self.parameters['control_type'] == 'joint_torque':

            print("Not implemented")

        elif self.parameters['control_type'] == 'joint_position':

            print("Not implemented")

        else:

            raise Exception('[Error] Unknown control type for bullet environment initialized')

        #Run the simulator
        self.world.run_simulation(control_info['no_samples'])

    # Internal function to read the values of the parameter variables from the bullet simulation environment
    # in non realtime case
    def _read_params_nrbullet(self):

        params_val = {}
        parameters = self.parameters

        for params_name in self.params_names:

            param_val = []
            param_info = parameters['params'][params_name]

            if param_info['type'] == 'joint_position':

                
                jointsInfo = self.world.readJointState(param_info['robotID'], param_info['joint_indices'])
                for jointInfo in jointsInfo:
                    param_val.append(jointInfo[0])

                params_val[params_name] = param_val

            elif param_info['type'] == 'joint_velocity':

                jointsInfo = self.world.readJointState(param_info['robotID'], param_info['joint_indices'])
                for jointInfo in jointsInfo:
                    param_val.append(jointInfo[1])

                params_val[params_name] = param_val


            elif param_info['type'] == 'joint_torque':

                jointsInfo = self.world.readJointState(param_info['robotID'], param_info['joint_indices'])
                for jointInfo in jointsInfo:
                    param_val.append(jointInfo[3])

                params_val[params_name] = param_val

            elif param_info['type'] == 'progress_variable':
                if self.mpc_ran:
                    if 'state' in param_info:

                        params_val[params_name] = self.sol_mpc[0][params_name[0:-1]][1]

                    elif 'control' in param_info:

                        params_val[params_name] = self.sol_mpc[1][params_name[0:-1]][1]
                else:

                    params_val[params_name] = 0


            else:

                print("[ERROR] Invalid type of parameter to be read from the simulation environment")


        return params_val


# TODO: set a method to let the user define the inputs and outputs of the function get from opti.to_function
# TODO: This should also account for monitors
# TODO: Set offline solution for initialization ( mpc.set_offline_solution(solver, options, ?))

if __name__ == '__main__':

    print("No syntax errors")
