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
        self.type = sim_type
        self.parameters = parameters
        #Create a list of the MPC states, variables, controls and parameters in a fixed order
        self.params_names = tc.parameters.keys() 
        self.states_names = tc.states.keys()
        self.controls_names = tc.controls.keys()
        self.variables_names = tc.variables.keys()

        self.generated_functions = {} #add all the casadi functions for later saving as .casadi files
        self._create_ocp_vars()

    def create_OCP_binary(self):

        ''' 
        Creates a casadi function for solving the OCP using IPOPT solver to a high degree of tolerance
        '''
        tc = self.tc
        tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
        ocp_fun = tc.ocp.opti.to_function('ocp_fun', self.opti_xplam, opti_xplam)
        self.generated_functions['ocp_fun'] = ocp_fun

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