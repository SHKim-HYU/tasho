# Simulates the MPC motion skill designed using Tasho

import casadi as cs
import numpy as np
from time import time
import json, math


class MPC:
    

    def __init__(self, name, json_file):

        """
        Creates an object that simulates the MPC controller, which assumes a full state feedback. It is designed to be very similar to the
        Orocos component used to deploy the Tasho controllers on the hardware. It is not real-time. Does not
        run on a separate thread/process compared to the main program. The main purpose of this object is simulation.

        MPC component is defined only for fixed-time single-stage OCP problems.
        """

        # Create a list of the MPC states, variables, controls and parameters in a fixed order
        self.properties = {}
        self.input_ports = {}
        self.output_ports = {}
        self.event_input_port = {}
        self.event_output_port = {}
        self.max_mpc_iter = 1000  # limit on the number of MPC iterations

        # Readint the json file
        with open('json_file') as F:
            json_dict = json.load(F)
        self.json_dict = json_dict

        # initializing common properties
        self.horizon = json_dict["horizon"]
        self.sampling_time = json_dict["sampling_time"]
        self.ocp_file = json_dict["ocp_file"]
        self.mpc_file = json_dict["mpc_file"]
        self.pred_file = json_dict["pred_file"]
        self.num_states = json_dict['num_states']
        self.num_controls = json_dict['num_controls']
        self.num_properties = json_dict['num_properties']
        self.num_inp_ports = json_dict['num_inp_ports']
        self.num_out_ports = json_dict['num_out_ports']

        if "max_iter" in json_dict:
            self.max_mpc_iter = json_dict["max_mpc_iter"]

        # create the properties
        for i in range(self.num_properties):
            prop = json_dict["props"][i]
            self.properties[prop["name"]] = {'val':[], 'desc':prop["desc"]}

        # creating the input ports
        for i in range(self.num_inp_ports):
            in_port = json_dict["inp_ports"][i]
            self.input_ports[in_port["name"]] = {'val':[], 'desc':in_port['desc']}

        # creating the output ports
        for i in range(self.num_out_ports):
            out_port = json_dict["out_port"]
            self.output_ports[out_port["name"]] = {'val':[], 'desc':out_port['desc']}

        # load the casadi OCP function
        self.ocp_fun = self.load_casadi_fun(self.ocp_file, json_dict['ocp_fun_name'])

        # load the casadi MPC function if different from the OCP function
        if self.ocp_file != self.mpc_file:
            self.mpc_fun = self.load_casadi_fun(self.mpc_file, json_dict['mpc_fun_name'])
        else:
            self.mpc_fun = self.ocp_fun

        # load the prediction function to simulate the dynamics of the system
        self.pred_fun = self.load_casadi_fun(self.pred_file, json_dict['pred_fun_name'])
        self.variables_optiform_details = {}

        # creating the vector that will be the input to the CasADi functions
        self.x_vals = cs.DM.zeros(self.ocp_fun.nnz_in)
        self.res_vals = cs.DM.zeros(self.ocp_fun.nnz_out)

        self.states_log = []
        self.controls_log = []
        self.variables_log = []

        self._opti_xplam = (
            []
        )  # list of all the decision variables, parameters and lagrange multipliers
        # casadi function to take list of variables and convert to opti.x and opti.p form
        self._list_to_optip_fun = None
        self._statecontrolvariablelist_to_optix_fun = None
        # casadi function to take opti.x and opti.p and convert to list of states, controls and variables
        self._optip_to_paramlist = None
        self._optix_to_statecontrolvariablelist = None
        self._solver_time = (
            []
        )  # array to keep track of the time taken by the solver in every MPC step
        

    def load_casadi_fun(casadi_file, cas_fun_name):
        """
        Loads serialized or code-generated CasADi function.

        :param casadi_file: The location of the file from which the CasADi function must be loaded.
        :type casadi_file: String

        :param cas_fun_name: (optional) The name of the CasADi function that must be loaded. Must be specified for a .so file.
        """

        # loads a .casadi function, the easiest form.
        if ".casadi" in casadi_file:
            return cs.Function.load(casadi_file)
        
        # loads .so file, that is obtained by compiling a code-generated CasADi file
        elif ".so" in casadi_file:
            return cs.external(cas_fun_name, casadi_file)

        else:
            raise Exception("CasADi file of unknown format provided")


    ## Configures the MPC from the current positions
    def configMPC(self):

        """
        Configures the MPC solver, solves the OCP and warmstarts the MPC. The properties and ports of the
        MPC object must be connected and initialized connected before calling configMPC().
        """

        # read the properties
        self._read_properties()
        
        # read the input ports
        self._read_input_ports()

        # Evaluate the OCP function
        self.res_val = self.ocp_fun(self.x_vals)

        # if the mpc solver is different from the OCP solver, evaluate it with the solution to warmstart
        if self.ocp_file != self.mpc_file:
            self.res_val = self.mpc_fun(self.res_val[0:self.x_vals.shape[0]])

        # configure the solver for the MPC iterations
        if self.mpc_debug is not True:
            self._create_mpc_fun_casadi(codeGen=self.codeGen, f_name="ocp_fun")

        # if self.parameters["codegen"]["compilation"]:
        #     import os
        #     if self.parameters["codegen"]["compiler"]:
        #         compiler = self.parameters["codegen"]["compiler"]
        #     else:
        #         compiler = "gcc"
        #     print("Compiling MPC function ...")
        #     os.system(
        #         compiler
        #         + " "
        #         + self.parameters["codegen"]["flags"]
        #         + " "
        #         + filename
        #         + ".c -shared -fPIC -lm -o "
        #         + filename
        #         + ".so"
        #     )
        #     print("... Finished compilation of MPC function")
        # if self.parameters["codegen"]["jit"]:
        #     print("Using just-in-time compilation ...")
            # cg_opts = {"jit":True, "compiler": "shell", "jit_options": {"verbose":True, "compiler": "ccache gcc" , "compiler_flags": self.parameters["codegen"]["flags"]}, "verbose":False, "jit_serialize": "embed"}

    def _shift_states_and_controls(self):

        """
        Shifts all the states and controls by one step for warmstarting the OCP
        """
        
        horizon = self.horizon
        json_dict = self.json_dict
        
        # concatenate the control and the state variables for warmstarting
        vars = json_dict["states"]  + json_dict["controls"]
        x_vals = self.x_vals
        res_vals = self.res_vals

        # For states, shift its values by backward by one step but leave the first state untouched
        # because it is updated by sensor readings

        for v in json_dict["states"]:
            start = json_dict[v]["start"]
            jump = json_dict[v]["jump"]
            x_vals[start + jump : json_dict[v]["end"] - jump] = res_vals[start + 2*jump : json_dict[v]["end"]]

        # For controls, shift its values backward by one step for warmstarting
        for v in json_dict["controls"]:
            start = json_dict[v]["start"]
            jump = json_dict[v]["jump"]
            x_vals[start: json_dict[v]["end"] - jump] = res_vals[start + jump : json_dict[v]["end"]]

        # # For all states, simply copy the terminal state from the result. A better strategy is perhaps to simulate, but 
        # # this is also effective
        # for s in json_dict["states"]:
        #     x_vals[json_dict[s]["end"] - json_dict[s]["jump"] : json_dict[s]["end"]] = res_vals[
        #         json_dict[s]["end"] - json_dict[s]["jump"] : json_dict[s]["end"]]
                

    def _sim_dynamics_update(self):
        """
        Internal function to simulate the system by one MPC time step to predict the state
        from which the next control output will be applied. A practical method to deal with the MPC
        computation delay. 

        It should be run before shifting the states and controls.
        """

        json_dict = self.json_dict
        X = []
        # concatenate the initial states
        for var in json_dict["states"]:
            X = cs.vertcat(X, self._get_initial_state_control(var))

        # concatenate initial controls
        U = []
        for var in json_dict["controls"]:
            U = cs.vertcat(U, self._get_initial_state_control(var))

        # concatenate initial parameters
        P = []
        for var in json_dict["parameters"]:
            P = cs.vertcat(P, self._get_initial_state_control(var))

        self.X_new = self.pred_fun(X, U, P, self.sampling_time)

    # Continuous running of the MPC
    def runMPC(self):

        sol_states = self.sol_ocp[0]
        sol_controls = self.sol_ocp[1]
        sol_variables = self.sol_ocp[2]
        control_info = self.parameters["control_info"]
        tc = self.tc
        sol = [0] * len(self._opti_xplam)
        par_start_element = (
            len(self.states_names)
            + len(self.controls_names)
            + len(self.variables_names)
        )

        # TODO: change by adding termination criteria
        for mpc_iter in range(self.max_mpc_iter):

            if self.type == "bullet_notrealtime":

                # reading and setting the latest parameter values and applying MPC action
                params_val = self._read_params_nrbullet()
                if mpc_iter == 20 and self.perturb:
                    q_perturbed = params_val["q0"]
                    q_perturbed[5] += 0.2
                    joint_indices = [11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7]
                    # print(q_perturbed)
                    self.world.resetJointState(
                        control_info["robotID"], joint_indices, q_perturbed
                    )
                    # params_val = self._read_params_nrbullet()

                # if self.mpc_ran:
                # for param in params_val:
                #     print("Abs error in "+ param)
                #     print(cs.fabs(cs.vec(old_params_val[param]) - cs.vec(params_val[param])))
                sol_mpc = [sol_states, sol_controls, sol_variables]
                self.sol_mpc = sol_mpc
                self._apply_control_nrbullet(sol_mpc, params_val)
                self.mpc_ran = True

                # simulate to predict the future state when the first control input is applied
                # to use that as the starting state for the MPC and accordingly update the params_val
                self._sim_dynamics_update_params(params_val, sol_states, sol_controls)

                # When the mpc_fun is not initialized as codegen or .casadi function
                if self._mpc_fun == None:
                    for params_name in self.params_names:
                        tc.ocp.set_value(
                            tc.parameters[params_name], params_val[params_name]
                        )
                    # set the states, controls and variables as initial values
                    self._warm_start(
                        [sol_states, sol_controls, sol_variables], options="shift"
                    )
                    try:
                        sol = tc.solve_ocp()
                    except:
                        tc.ocp.show_infeasibilities(0.5 * 1e-6)
                        raise Exception("Solver crashed")

                else:
                    # print("before calling printing system dynamics function")
                    self._warm_start_casfun(
                        [sol_states, sol_controls, sol_variables], sol, options="shift"
                    )
                    # print("this ran")
                    # print(sol_variables)
                    i = par_start_element
                    for params_name in self.params_names:
                        sol[i] = params_val[params_name]
                        i += 1
                    tic = time()
                    sol = list(self._mpc_fun(*sol))
                    toc = time() - tic
                    self._solver_time.append(toc)

                    # Monitors
                    opti_form = self._opti_xplam_to_optiform(*sol)
                    # computing the primal feasibility of the solution
                    fun_pr = tc.function_primal_residual()
                    residual_max = fun_pr(*opti_form)
                    print("primal residual is : " + str(residual_max.full()))

                    if residual_max.full() >= 1e-3:
                        print("Solver infeasible")
                        return "MPC_FAILED"

                    # checking the termination criteria

                    # print(opti_form)
                    print(tc.monitors["termination_criteria"]["monitor_fun"](opti_form))
                    if tc.monitors["termination_criteria"]["monitor_fun"](opti_form):
                        print(self._solver_time)
                        print(
                            "MPC termination criteria reached after "
                            + str(mpc_iter)
                            + " number of MPC samples. Exiting MPC loop."
                        )

                        self.world.setController(
                            control_info["robotID"],
                            "velocity",
                            control_info["joint_indices"],
                            targetVelocities=[0] * len(control_info["joint_indices"]),
                        )
                        return "MPC_SUCCEEDED"

                sol_states, sol_controls, sol_variables = self._read_solveroutput(sol)

                # Log solution
                if ("log_solution" in self.parameters) and self.parameters["log_solution"]:
                    self.states_log.append(sol_states)
                    self.controls_log.append(sol_controls)
                    self.variables_log.append(sol_variables)

                self.mpc_ran = True

                old_params_val = params_val  # to debug how the prediction varies from the actual plant after one step of
                # control is applied
                # Apply the control action to bullet environment
                # self._apply_control_nrbullet(sol_mpc) #uncomment if the simulation of system to predict future is not done

            elif self.type == "bullet_realtime":

                print("Not implemented")

            else:

                print("[ERROR] Unknown simulation type")

            if mpc_iter == self.max_mpc_iter - 1:
                print("MPC timeout")
                print(self._solver_time)
                return "MPC_TIMEOUT"

    
    def _read_properties(self):

        """
        Reads properties and assigns them to the correct location of the input vector of the MPC function
        """
        json_dict = self.json_dict
        for i in range(self.num_properties):
            prop = json_dict["props"][i]
            self.x_vals[json_dict[prop["var"]]["start"] : json_dict[prop["var"]["end"]]] = self.properties[prop["name"]]["val"]

    def _read_input_ports(self):

        """
        Reads the values supplied to the input ports of the MPC. And assigns it to the correct location of the MPC input vector.
        """

        json_dict = self.json_dict
        for i in range(self.num_inp_ports):
            port = json_dict["inp_ports"][i]
            start = json_dict[port["var"]]["start"]
            size = json_dict[port["var"]]["size"]
            self.x_vals[start : start + size] = self.input_ports[port["name"]]["val"]

    def _write_output_ports(self, sequence):

        """
        Writes the solution of the OCP/MPC to the relevant output ports. 

        :param sequence: The time instant in the horizon whose values must be written into the port.
        :type sequence: int
        """
        
        json_dict = self.json_dict
        for i in range(self.num_out_ports):
            port = json_dict["out_ports"][i]
            var = json_dict[port["name"]]["var"]
            start = var["start"]
            jump = var["jump"]
            self.output_ports[port["name"]]["val"] = self.res_vals[start + sequence*jump : start + sequence*jump  + var["size"]]

    def _get_initial_state_control(self, var, x_vals):

        """
        Returns the value at the start of the horizon of the given variable from the given array.

        :param var: the variable whose initial value is returned
        :type var: String

        :param x_vals: The input/output vector of OCP/MPC
        :type x_vals: Casadi DM vector
        """
        
        json_dict = self.json_dict
        return x_vals[json_dict["var"]["start"] : json_dict["var"]["start"] + json_dict["var"]["size"]]

# TODO: set a method to let the user define the inputs and outputs of the function get from opti.to_function
# TODO: This should also account for monitors
# TODO: Set offline solution for initialization ( mpc.set_offline_solution(solver, options, ?))

if __name__ == "__main__":

    print("No syntax errors")
