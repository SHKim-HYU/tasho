# A function to save the casadi binaries for all the relevant actions 
# to be later called from orocos/ros functions

import casadi as cs
from tasho import task_prototype_rockit as tp

class tp_to_casadi_binaries:
    ''' tp_to_casadi_binaries(tc, type)
    
    :param tc: task context defined using the task protype function
    
    :param parameters: Extra details for the MPC. Such as the mapping between the parameter variables and the data from bullet

    '''

    def __init__(self, tc, parameters):


        self.tc = tc
        self.parameters = parameters
        #Create a list of the MPC states, variables, controls and parameters in a fixed order
        self.params_names = tc.parameters.keys() 
        self.states_names = tc.states.keys()
        self.controls_names = tc.controls.keys()
        self.variables_names = tc.variables.keys()

        self.generated_functions = {} #add all the casadi functions for later saving as .casadi files
        self._create_ocp_vars()
        self._create_OCP_binary()
        self._create_MPC_binary() #todo: add a condition to check if the MPC is optional

    def _create_OCP_binary(self):

        tc = self.tc
        tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
        ocp_fun = tc.ocp.opti.to_function('ocp_fun', self._ocp_vars, self._ocp_vars)
        self.generated_functions['ocp_fun'] = ocp_fun

    def _create_MPC_binary(self):

        tc = self.tc
        #configure the solver for the MPC iterations
        if self.parameters['solver_name'] == 'ipopt':

            if 'lbfgs' in self.parameters['solver_params']:

                tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3, 'print_level':5}})

        elif self.parameters['solver_name'] == 'sqpmethod':

            if 'qrqp' in self.parameters['solver_params']:
                kkt_tol_pr = 1e-3
                kkt_tol_du = 1e-1
                min_step_size = 1e-6
                max_iter = 2
                max_iter_ls = 3
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
                ipopt_max_iter = 50
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

        #create the MPC fun
        mpc_fun = tc.ocp.opti.to_function('mpc_fun', self._ocp_vars, self._ocp_vars)
        self.generated_functions['mpc_fun'] = mpc_fun


    #internal function to create the OCP/MPC function using casadi's opti.to_function capability
    def _create_ocp_vars(self):

        tc = self.tc
        #create a list of ocp decision variables in the following order
        #states, controls, variables, parameters, lagrange_multipliers
        opti_xplam = []
        for state in self.states_names:
            _, temp = tc.ocp.sample(tc.states[state], grid = 'control')
            opti_xplam.append(temp)

        #obtain the opti variables related to control
        for control in self.controls_names:
            _, temp = tc.ocp.sample(tc.controls[control], grid = 'control')
            temp2 = []
            for i in range(temp.shape[1] - 1):
                temp2.append(tc.ocp._method.eval_at_control(tc.ocp, tc.controls[control], i))
            opti_xplam.append(cs.hcat(temp2))

        #obtain the opti variables related for variables
        for variable in self.variables_names:
            temp = tc.ocp._method.eval_at_control(tc.ocp, tc.variables[variable], 0)
            print(temp)
            opti_xplam.append(temp)

        for parameter in self.params_names:
            temp = tc.ocp._method.eval_at_control(tc.ocp, tc.parameters[parameter], 0)#tc.ocp.sample(tc.parameters[parameter])
            print(temp)
            opti_xplam.append(temp)

        #adding the lagrange multiplier terms as well
        # print(tc.ocp.opti.p)
        # print(opti_xplam[3])
        # print(opti_xplam[4])
        # print(opti_xplam[5])
        # print(opti_xplam[6])
        opti_xplam.append(tc.ocp.opti.lam_g)

        #setting the MPC function!
        #opti = tc.ocp.opti
        #self._mpc_fun = tc.ocp.opti.to_function('mpc_fun', [opti.p, opti.x, opti.lam_g], [opti.x, opti.lam_g, opti.f]);
        # self._mpc_fun = tc.ocp.opti.to_function('mpc_fun', opti_xplam, opti_xplam)
        self._ocp_vars = opti_xplam
        opti = tc.ocp.opti
        self._ocp_vars_to_optiform = cs.Function("ocp_vars_to_optiform", opti_xplam, [opti.x, opti.p, opti.lam_g])

    def save_all_functions(self, destination):
        '''
        Saves all the created casadi functions in the form of .casadi files in the specified destination
        '''

        print("Not implemented")



if __name__ == '__main__':

    print("No syntax errors")