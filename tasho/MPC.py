# Simulates the MPC motion skill designed using Tasho

import casadi as cs
import numpy as np
from time import time
import json


class MPC:
    

    def __init__(self, name, json_file):

        """
        Creates an object that simulates the MPC controller. It is designed to be very similar to the
        Orocos component used to deploy the Tasho controllers on the hardware. It is not real-time. Does not
        run on a separate thread/process compared to the main program. The main purpose of this object is simulation.
        """

        # Create a list of the MPC states, variables, controls and parameters in a fixed order
        self.params_names = tc.parameters.keys()
        self.states_names = tc.states.keys()
        self.controls_names = tc.controls.keys()
        self.variables_names = tc.variables.keys()
        self.variables_optiform_details = {}

        self.params_history = []

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
        self.max_mpc_iter = 100  # limit on the number of MPC iterations

    ## Configures the MPC from the current positions
    def configMPC(self, init_guess=None):

        # SOLVE the OCP in order to warm start the MPC

        tc = self.tc
        self.mpc_ran = False
        params_val = self._read_params_nrbullet()

        # set the parameter values
        for params_name in self.params_names:

            tc.set_value(tc.parameters[params_name], params_val[params_name])

        # set the initial guesses

        sol = tc.solve_ocp()
        # print(tc.ocp._method.N)
        sol_states, sol_controls, sol_variables = self._read_solveroutput(sol)
        self.sol_ocp = [sol_states, sol_controls, sol_variables]

        # configure the solver for the MPC iterations

        self._warm_start(self.sol_ocp)
        sol = tc.solve_ocp()
        if self.mpc_debug is not True:
            self._create_mpc_fun_casadi(codeGen=self.codeGen, f_name="ocp_fun")

            self._warm_start(self.sol_ocp)
            print("Solving with the SQP method")
            sol = tc.solve_ocp()

            if self.mpc_debug is not True:
                if ("codegen" in self.parameters) and self.parameters["codegen"]:
                    self.code_type = 2

                    if self.parameters["codegen"]["filename"]:
                        filename = self.parameters["codegen"]["filename"]
                    else:
                        filename = "mpc_fun"

                    self._create_mpc_fun_casadi(
                        codeGen=self.parameters["codegen"]["codegen"],
                        f_name=filename,
                    )

                    if self.parameters["codegen"]["compilation"]:
                        import os

                        if self.parameters["codegen"]["compiler"]:
                            compiler = self.parameters["codegen"]["compiler"]
                        else:
                            compiler = "gcc"

                        print("Compiling MPC function ...")
                        os.system(
                            compiler
                            + " "
                            + self.parameters["codegen"]["flags"]
                            + " "
                            + filename
                            + ".c -shared -fPIC -lm -o "
                            + filename
                            + ".so"
                        )
                        print("... Finished compilation of MPC function")

                    if self.parameters["codegen"]["use_external"]:

                        import os

                        if os.path.isfile(filename+'.so'):
                            print("Loading the compiled function back ...")

                            self._mpc_fun = cs.external(
                                filename, "./" + filename + ".so"
                            )

                            print("... Loaded")
                        else:
                            print("[ERROR] The file "+filename+".so doesn't exist. Try compiling the function first")
                            exit()

        self.system_dynamics = self.tc.stages[0]._method.discrete_system(self.tc.stages[0])

    # internal function to create the MPC function using casadi's opti.to_function capability
    def _create_mpc_fun_casadi(
        self, codeGen=False, f_name="mpc_fun", codeGen_dest="./"
    ):

        tc = self.tc
        # create a list of ocp decision variables in the following order
        # states, controls, variables, parameters, lagrange_multipliers
        opti_xplam = []
        opti_xplam_cg = []
        vars_db = self.variables_optiform_details
        counter = 0
        print(counter)
        for state in self.states_names:
            _, temp = tc.stages[0].sample(tc.states[state], grid="control")
            temp2 = []
            for i in range(tc.horizon + 1):
                temp2.append(temp[:, i])
            opti_xplam.append(temp)
            vars_db[state] = {"start": counter, "size": temp.shape[0]}
            opti_xplam_cg.append(cs.vcat(temp2))
            counter += temp.shape[0] * temp.shape[1]
            vars_db[state]["end"] = counter

        # obtain the opti variables related to control
        for control in self.controls_names:
            _, temp = tc.stages[0].sample(tc.controls[control], grid="control")
            temp2 = []
            for i in range(tc.horizon):
                temp2.append(
                    tc.stages[0]._method.eval_at_control(tc.stages[0], tc.controls[control], i)
                )
            vars_db[control] = {"start": counter, "size": temp.shape[0]}
            counter += temp.shape[0] * (temp.shape[1] - 1)
            vars_db[control]["end"] = counter
            opti_xplam.append(cs.hcat(temp2))
            opti_xplam_cg.append(cs.vcat(temp2))

        # obtain the opti variables related for variables
        for variable in self.variables_names:
            temp = tc.stages[0]._method.eval_at_control(tc.stages[0], tc.variables[variable], 0)
            # print(temp)
            opti_xplam.append(temp)
            opti_xplam_cg.append(temp)
            vars_db[variable] = {"start": counter, "size": temp.shape[0]}
            counter += temp.shape[0]
            vars_db[variable]["end"] = counter

        for parameter in self.params_names:
            temp = tc.stages[0]._method.eval_at_control(
                tc.stages[0], tc.parameters[parameter], 0
            )  # tc.ocp.sample(tc.parameters[parameter])
            # print(temp)
            opti_xplam.append(temp)
            opti_xplam_cg.append(temp)
            vars_db[parameter] = {"start": counter, "size": temp.shape[0]}
            counter += temp.shape[0]
            vars_db[parameter]["end"] = counter

        # adding the lagrange multiplier terms as well
        # print(tc.ocp.opti.p)
        # print(opti_xplam[3])
        # print(opti_xplam[4])
        # print(opti_xplam[5])
        # print(opti_xplam[6])
        opti_xplam.append(tc.ocp._method.opti.lam_g)
        vars_db["lam_g"] = {
            "start": counter,
            "size": tc.ocp._method.opti.lam_g.shape[0],
        }
        counter += tc.ocp._method.opti.lam_g.shape[0]
        vars_db["lam_g"]["end"] = counter
        vars_db["horizon"] = tc.horizon
        vars_db["ocp_rate"] = tc.ocp_rate
        # setting the MPC function!
        # opti = tc.ocp.opti
        opti = tc.ocp._method.opti
        # self._mpc_fun = opti.to_function(
        #     "mpc_fun", [opti.p, opti.x, opti.lam_g], [opti.x, opti.lam_g, opti.f]
        # )
        if self.parameters["codegen"]["jit"]:
            print("Using just-in-time compilation ...")
            cg_opts = {"jit":True, "compiler": "shell", "jit_options": {"verbose":True, "compiler": "ccache gcc" , "compiler_flags": self.parameters["codegen"]["flags"]}, "verbose":False, "jit_serialize": "embed"}

            self._mpc_fun = tc.ocp.to_function(f_name, opti_xplam, opti_xplam, cg_opts)
            self._mpc_fun_cg = tc.ocp.to_function(
                f_name, [cs.vcat(opti_xplam_cg)], [cs.vcat(opti_xplam_cg)], cg_opts
            )
        else:
            self._mpc_fun = tc.ocp.to_function(f_name, opti_xplam, opti_xplam)
            self._mpc_fun_cg = tc.ocp.to_function(
                f_name, [cs.vcat(opti_xplam_cg)], [cs.vcat(opti_xplam_cg)]
            )
        self._opti_xplam = opti_xplam

        self._opti_xplam_to_optiform = cs.Function(
            "opti_xplam_to_optiform", opti_xplam, [opti.x, opti.p, opti.lam_g]
        )
        if codeGen:
            if self.code_type == 0:
                self._mpc_fun_cg.generate(
                    f_name + ".c", {"with_header": True, "main": True}
                )
                C = cs.Importer(f_name + ".c", "clang")
                self._mpc_fun = cs.external("mpc_", C)
            elif self.code_type == 1:
                vars_db["casadi_fun"] = codeGen_dest + f_name + ".casadi"
                vars_db["fun_name"] = f_name
                print(vars_db)
                with open(codeGen_dest + f_name + "_property.json", "w") as fp:
                    json.dump(vars_db, fp)
                self._mpc_fun_cg.save(f_name + ".casadi")
            elif self.code_type == 2:
                self._mpc_fun_cg.generate(
                    f_name + ".c", {"with_header": True, "main": True}
                )
                vars_db["casadi_fun"] = codeGen_dest + f_name + ".casadi"
                vars_db["fun_name"] = f_name
                print(vars_db)
                with open(codeGen_dest + f_name + "_property.json", "w") as fp:
                #     json.dump(vars_db, fp)
                    json.dump(vars_db, fp, indent=2)
                self._mpc_fun_cg.save(codeGen_dest + f_name + ".casadi")
                # self._mpc_fun_cg.save(f_name + ".casadi")
            # self._mpc_fun = cs.external('f', './mpc_fun.so')

    ## obtain the solution of the ocp
    def _read_solveroutput(self, sol):

        sol_states = {}
        sol_controls = {}
        sol_variables = {}
        tc = self.tc

        if self._mpc_fun == None:
            for state in tc.states:
                _, sol_state = sol(tc.stages[0]).sample(tc.states[state], grid="control")
                sol_states[state] = sol_state

            for control in tc.controls:
                _, sol_control = sol(tc.stages[0]).sample(tc.controls[control], grid="control")
                sol_controls[control] = sol_control[0:-1]

            for variable in tc.variables:
                _, sol_variable = sol(tc.stages[0]).sample(tc.variables[variable], grid="control")
                sol_variable = sol_variable[0]
                sol_variables[variable] = sol_variable

        # when the solver directly gives the list of decision variables
        else:
            i = 0
            for state in self.states_names:
                # print("Shape of state " + state + " is = ")
                # print(sol[i].shape)
                sol_states[state] = np.array(sol[i].T)
                i += 1

            for control in tc.controls:
                sol_controls[control] = np.array(sol[i].T)
                i += 1

            for variable in tc.variables:
                sol_variables[variable] = np.array(sol[i].T)
                i += 1

        return sol_states, sol_controls, sol_variables

    ## Function to provide the initial guess for warm starting the states, controls and variables in tc
    def _warm_start(self, sol_ocp, options="reuse"):

        tc = self.tc
        # reusing the solution from previous MPC iteration for warm starting
        if self.mpc_ran == False or options == "reuse":

            sol_states = sol_ocp[0]
            for state in tc.states:
                tc.set_initial(tc.states[state], sol_states[state].T)

            sol_controls = sol_ocp[1]
            for control in tc.controls:
                tc.set_initial(tc.controls[control], sol_controls[control].T)

            sol_variables = sol_ocp[2]
            for variable in tc.variables:
                tc.set_initial(tc.variables[variable], sol_variables[variable])

        # warm starting by shiting the solution by 1 step
        elif options == "shift":

            sol_states = sol_ocp[0]
            for state in tc.states:
                # print(state)
                # print(sol_states[state][1:].shape)
                # print(sol_states[state][-1:].shape)
                tc.set_initial(
                    tc.states[state],
                    cs.vertcat(sol_states[state][1:], sol_states[state][-1:]).T,
                )
                # tc.ocp.set_initial(tc.states[state], sol_states[state].T)

            sol_controls = sol_ocp[1]
            for control in tc.controls:
                # tc.ocp.set_initial(tc.controls[control], sol_controls[control].T)
                # print(control)
                # print(sol_controls[control][1:-1].shape)
                zeros_np = np.zeros(sol_controls[control][-1:].shape)
                # print(zeros_np.shape)
                tc.set_initial(
                    tc.controls[control],
                    cs.vertcat(sol_controls[control][1:-1], zeros_np, zeros_np).T,
                )

            sol_variables = sol_ocp[2]
            for variable in tc.variables:
                tc.set_initial(tc.variables[variable], sol_variables[variable])

        else:

            raise Exception("Invalid MPC restart option " + options)

    ## Function to provide the initial guess for warm starting the states, controls and variables in tc
    # in a format that conforms to the inputs of _mpc_fun when .casadi or codegen is available
    def _warm_start_casfun(self, sol_ocp, sol, options="reuse"):
        # TODO: refactor this to completely do away with sol_ocp
        tc = self.tc
        i = 0
        # reusing the solution from previous MPC iteration for warm starting
        if self.mpc_ran == False or options == "reuse":

            sol_states = sol_ocp[0]
            for state in tc.states:
                sol[i] = sol_states[state].T
                i += 1

            sol_controls = sol_ocp[1]
            for control in tc.controls:
                sol[i] = sol_controls[control][0:-1].T
                i += 1

            sol_variables = sol_ocp[2]
            for variable in tc.variables:
                sol[i] = sol_variables[variable].T
                i += 1

        # warm starting by shiting the solution by 1 step
        elif options == "shift":

            sol_states = sol_ocp[0]
            for state in tc.states:
                sol[i] = cs.vertcat(sol_states[state][1:], sol_states[state][-1:]).T
                i += 1

            sol_controls = sol_ocp[1]
            for control in tc.controls:
                zeros_np = np.zeros(sol_controls[control][-1:].shape)
                sol[i] = cs.vertcat(sol_controls[control][1:], zeros_np).T
                i += 1

            sol_variables = sol_ocp[2]
            for variable in tc.variables:
                sol[i] = sol_variables[variable]
                i += 1

        else:

            raise Exception("Invalid MPC restart option " + options)

    def _sim_dynamics_update_params(self, params_val, sol_states, sol_controls):
        """
        Internal function to simulate the dynamics by one time step to predict the states of the MPC
        when the first control input is applied.
        """
        print("Before printing system dynamics")
        print(params_val)
        # obtain the current state of the system from params and the first control input
        X = []
        U = []
        control_info = self.parameters["control_info"]
        for state in self.states_names:
            X.append(params_val[state + "0"])
        for control in self.controls_names:
            U.append(sol_controls[control][0])
        X = cs.vcat(X)
        U = cs.vcat(U)
        print("Ushape")
        print(U.shape)

        # simulate the system dynamics to obtain the future state
        next_X = self.system_dynamics(x0=X, u=U, T=self.parameters["t_mpc"])["xf"]
        # update the params_val with this info
        start = 0
        for state in self.states_names:
            print(state)
            state_shape = self.tc.states[state].shape
            state_len = state_shape[0] * state_shape[1]
            if state == "q_dot" and self.parameters["control_type"] == "joint_velocity":
                # print(state)
                # print(np.array(params_val[state+'0']).T)
                # print(np.array(next_X[start:start+state_len]))
                # params_val[state + "0"] = (
                #     0.5
                #     * (
                #         np.array(params_val[state + "0"])
                #         + np.array(next_X[start : start + state_len].T)
                #     ).T
                # )
                params_val[state + "0"] = (
                    np.array(params_val[state + "0"]) * (1 / control_info["no_samples"])
                    + np.array(next_X[start : start + state_len].T)
                    * (1 - 1 / control_info["no_samples"])
                ).T
            else:
                params_val[state + "0"] = np.array(next_X[start : start + state_len])
            # print(params_val[state+"0"])
            start = state_len + start

        # print("After printing system dynamics")
        # print(params_val)

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
                    joint_indices = [
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                    ]
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


# TODO: set a method to let the user define the inputs and outputs of the function get from opti.to_function
# TODO: This should also account for monitors
# TODO: Set offline solution for initialization ( mpc.set_offline_solution(solver, options, ?))

if __name__ == "__main__":

    print("No syntax errors")
