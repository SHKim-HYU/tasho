# File to set the MPC options and to deploy it. Takes as input the task context.
# Provides options for code-generation for deployment on the robot
# Could also optionally create a rosnode that communicates with world_simulator for verifying the MPC in simulations
# Monitor the variables to raise events

import casadi as cs
import numpy as np
from time import time
from multiprocessing import Pool


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
        # Create a list of the MPC states, variables, controls and parameters in a fixed order
        self.params_names = tc.parameters.keys()
        self.states_names = tc.states.keys()
        self.controls_names = tc.controls.keys()
        self.variables_names = tc.variables.keys()
        self.variables_optiform_details = {}

        self.params_history = []

        # casadi function (could be codegenerated) to run the MPC (to avoid the preparation step)
        self.mpc_debug = False
        self._mpc_fun = None
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
        self.torque_effort_sumsqr = 0
        self.max_mpc_iter = 100  # limit on the number of MPC iterations
        self.codeGen = False
        self.perturb = False

        if sim_type == "bullet_notrealtime":

            self.world = parameters["world"]  # object of world_simulator class

        elif sim_type == "bullet_realtime":

            print("Not implemented")

        else:
            print("[ERROR] Unknown simulation type")

    ## Configures the MPC from the current positions
    def configMPC_fromcurrent(self, init_guess=None):

        # SOLVE the OCP in order to warm start the MPC

        tc = self.tc
        self.mpc_ran = False
        params_val = self._read_params_nrbullet()

        # set the parameter values
        for params_name in self.params_names:

            tc.ocp.set_value(tc.parameters[params_name], params_val[params_name])

        # set the initial guesses

        if init_guess != None:

            print("Not implemented")

        # For initial guesses, setting a robust solver (IPOPT)
        tc.set_ocp_solver(
            "ipopt",
            {
                "ipopt": {
                    "max_iter": 1000,
                    "hessian_approximation": "limited-memory",
                    "limited_memory_max_history": 5,
                    "tol": 1e-3,
                }
            },
        )

        tc.set_ocp_solver(
            "ipopt",
            {
                "ipopt": {"max_iter": 1000, "tol": 1e-6, "linear_solver": "ma27"},
                "error_on_fail": True,
            },
        )

        # assuming that the discretization settings are already done!
        # print(tc.ocp._method.M)
        tc.set_discretization_settings(self.parameters["disc_settings"])

        sol = tc.solve_ocp()
        # print(tc.ocp._method.N)
        sol_states, sol_controls, sol_variables = self._read_solveroutput(sol)
        self.sol_ocp = [sol_states, sol_controls, sol_variables]

        # configure the solver for the MPC iterations
        if self.parameters["solver_name"] == "ipopt":

            if "lbfgs" in self.parameters["solver_params"]:

                tc.set_ocp_solver(
                    "ipopt",
                    {
                        "ipopt": {
                            "max_iter": 1000,
                            "hessian_approximation": "limited-memory",
                            "limited_memory_max_history": 5,
                            "tol": 1e-3,
                            "print_level": 5,
                        },
                        "error_on_fail": True,
                    },
                )
            else:
                tc.set_ocp_solver(
                    "ipopt",
                    {
                        "ipopt": {
                            "max_iter": 100,
                            "tol": 1e-3,
                            "mu_init": 1e-3,
                            "linear_solver": "ma97",
                            "fixed_variable_treatment": "make_parameter",
                            "hessian_constant": "yes",
                            # "jac_c_constant": "yes",
                            # "jac_d_constant": "yes",
                            "accept_every_trial_step": "yes",
                            "print_level": 0,
                            "mu_strategy": "monotone",
                            "nlp_scaling_method": "none",
                            "check_derivatives_for_naninf": "no",
                            "ma97_scaling": "none",
                            "ma97_order": "amd",
                            "ma57_pivot_order": 0,
                            "warm_start_init_point": "yes",
                            "magic_steps": "yes",
                            "fast_step_computation": "yes",
                            "mu_allow_fast_monotone_decrease": "yes",
                            "ma27_skip_inertia_check": "yes",
                            "error_on_fail": True,
                            # "ma27_ignore_singularity": "yes",
                        }
                    },
                )
            tc.set_discretization_settings(self.parameters["disc_settings"])
            tc.ocp._method.main_transcribe(tc.ocp)
            self._warm_start(self.sol_ocp)
            sol = tc.solve_ocp()
            if self.mpc_debug is not True:
                self._create_mpc_fun_casadi(codeGen=self.codeGen, f_name="ocp_fun")

        elif self.parameters["solver_name"] == "sqpmethod":

            if "qrqp" in self.parameters["solver_params"]:
                kkt_tol_pr = 1e-3
                kkt_tol_du = 1e-1
                min_step_size = 1e-6
                max_iter = 5
                max_iter_ls = 3
                qpsol_options = {
                    "constr_viol_tol": kkt_tol_pr,
                    "dual_inf_tol": kkt_tol_du,
                    "verbose": False,
                    "print_iter": False,
                    "print_header": False,
                    "dump_in": False,
                    "error_on_fail": False,
                }
                solver_options = {
                    "qpsol": "qrqp",
                    "qpsol_options": qpsol_options,
                    "verbose": False,
                    "tol_pr": kkt_tol_pr,
                    "tol_du": kkt_tol_du,
                    "min_step_size": min_step_size,
                    "max_iter": max_iter,
                    "max_iter_ls": max_iter_ls,
                    "print_iteration": True,
                    "print_header": False,
                    "print_status": False,
                    "print_time": True,
                    "error_on_fail": True,
                }  # "convexify_strategy":"regularize"
                tc.set_ocp_solver("sqpmethod", solver_options)

            elif "osqp" in self.parameters["solver_params"]:
                kkt_tol_pr = 1e-3
                kkt_tol_du = 1e-1
                min_step_size = 1e-4
                max_iter = 5
                max_iter_ls = 2
                eps_abs = 1e-5
                eps_rel = 1e-5
                qpsol_options = {
                    "osqp": {
                        "alpha": 1,
                        "eps_abs": eps_abs,
                        "eps_rel": eps_rel,
                        "verbose": 0,
                    },
                    "dump_in": False,
                    "error_on_fail": False,
                }
                solver_options = {
                    "qpsol": "osqp",
                    "qpsol_options": qpsol_options,
                    "verbose": False,
                    "tol_pr": kkt_tol_pr,
                    "tol_du": kkt_tol_du,
                    "min_step_size": min_step_size,
                    "max_iter": max_iter,
                    "max_iter_ls": max_iter_ls,
                    "print_iteration": True,
                    "print_header": False,
                    "print_status": False,
                    "print_time": True,
                    "error_on_fail": True,
                }  # "convexify_strategy":"regularize"
                tc.set_ocp_solver("sqpmethod", solver_options)

            elif "qpoases" in self.parameters["solver_params"]:
                kkt_tol_pr = 1e-3
                kkt_tol_du = 1e-1
                min_step_size = 1e-4
                max_iter = 10
                max_iter_ls = 0
                qpoases_tol = 1e-4
                qpsol_options = {
                    "printLevel": "none",
                    "enableEqualities": True,
                    "initialStatusBounds": "inactive",
                    "terminationTolerance": qpoases_tol,
                }
                solver_options = {
                    "qpsol": "qpoases",
                    "qpsol_options": qpsol_options,
                    "verbose": False,
                    "tol_pr": kkt_tol_pr,
                    "tol_du": kkt_tol_du,
                    "min_step_size": min_step_size,
                    "max_iter": max_iter,
                    "max_iter_ls": max_iter_ls,
                    "print_iteration": True,
                    "print_header": False,
                    "print_status": False,
                    "print_time": True,
                    "error_on_fail": True,
                }  # "convexify_strategy":"regularize"
                tc.set_ocp_solver("sqpmethod", solver_options)

            elif "ipopt" in self.parameters["solver_params"]:
                kkt_tol_pr = 1e-3
                kkt_tol_du = 1e-1
                min_step_size = 1e-6
                max_iter = 5
                ipopt_max_iter = 20
                max_iter_ls = 0

                ipopt_tol = 1e-3
                tiny_step_tol = 1e-6
                mu_init = 0.001
                linear_solver = "ma97"
                # linear_solver = "mumps"

                ipopt_options = {
                    "tol": ipopt_tol,
                    "tiny_step_tol": tiny_step_tol,
                    "fixed_variable_treatment": "make_parameter",
                    "hessian_constant": "yes",
                    "jac_c_constant": "yes",
                    "jac_d_constant": "yes",
                    "accept_every_trial_step": "yes",
                    "mu_init": mu_init,
                    "print_level": 0,
                    "linear_solver": linear_solver,
                    # "mumps_mem_percent": 1000,
                    "mumps_pivtolmax": 1e-6,
                    # "mehrotra_algorithm": "no",
                    "mu_strategy": "monotone",
                    "nlp_scaling_method": "none",
                    "check_derivatives_for_naninf": "no",
                    "ma97_scaling": "none",
                    "ma97_order": "amd",
                    "ma57_pivot_order": 0,
                    "warm_start_init_point": "yes",
                    "magic_steps": "yes",
                    "fast_step_computation": "yes",
                    "mu_allow_fast_monotone_decrease": "yes",
                    "ma27_skip_inertia_check": "yes",
                    # "ma27_ignore_singularity": "yes",
                    # "honor_original_bounds": "no",
                    # "bound_mult_init_method": "constant",
                    # "mu_oracle": "loqo",
                    # "mu_linear_decrease_factor": 0.5,
                }
                nlpsol_options = {"ipopt": ipopt_options, "print_time": False}
                qpsol_options = {
                    "nlpsol": "ipopt",
                    "nlpsol_options": nlpsol_options,
                    "print_time": False,
                    "verbose": False,
                    "error_on_fail": False,
                }
                solver_options = {
                    "qpsol": "nlpsol",
                    "qpsol_options": qpsol_options,
                    "tol_pr": kkt_tol_pr,
                    "tol_du": kkt_tol_du,
                    "min_step_size": min_step_size,
                    "max_iter": max_iter,
                    "max_iter_ls": max_iter_ls,
                    "print_iteration": True,
                    "print_header": False,
                    "print_status": False,
                    "print_time": True,
                    "error_on_fail": True,
                }  # "convexify_strategy":"regularize"
                tc.set_ocp_solver("sqpmethod", solver_options)

            tc.set_discretization_settings(self.parameters["disc_settings"])
            tc.ocp._method.main_transcribe(tc.ocp)
            self._warm_start(self.sol_ocp)
            print("Solving with the SQP method")
            sol = tc.solve_ocp()

        else:
            # Set ipopt as default solver
            print("Using IPOPT with LBFGS as the default solver")
            tc.set_ocp_solver(
                "ipopt",
                {
                    "ipopt": {
                        "max_iter": 1000,
                        "hessian_approximation": "limited-memory",
                        "limited_memory_max_history": 5,
                        "tol": 1e-3,
                    }
                },
            )

        if self.mpc_debug is not True:
            self._create_mpc_fun_casadi(codeGen=self.codeGen)
        self.system_dynamics = self.tc.ocp._method.discrete_system(self.tc.ocp)
        # print(sol_controls['s_ddot'])

    # internal function to create the MPC function using casadi's opti.to_function capability
    def _create_mpc_fun_casadi(self, codeGen=False, f_name="mpc_fun"):

        tc = self.tc
        # create a list of ocp decision variables in the following order
        # states, controls, variables, parameters, lagrange_multipliers
        opti_xplam = []
        opti_xplam_cg = []
        vars_db = self.variables_optiform_details
        counter = 0
        print(counter)
        for state in self.states_names:
            _, temp = tc.ocp.sample(tc.states[state], grid="control")
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
            _, temp = tc.ocp.sample(tc.controls[control], grid="control")
            temp2 = []
            for i in range(tc.horizon):
                temp2.append(
                    tc.ocp._method.eval_at_control(tc.ocp, tc.controls[control], i)
                )
            vars_db[control] = {"start": counter, "size": temp.shape[0]}
            counter += temp.shape[0] * (temp.shape[1] - 1)
            vars_db[control]["end"] = counter
            opti_xplam.append(cs.hcat(temp2))
            opti_xplam_cg.append(cs.vcat(temp2))

        # obtain the opti variables related for variables
        for variable in self.variables_names:
            temp = tc.ocp._method.eval_at_control(tc.ocp, tc.variables[variable], 0)
            # print(temp)
            opti_xplam.append(temp)
            opti_xplam_cg.append(temp)
            vars_db[variable] = {"start": counter, "size": temp.shape[0]}
            counter += temp.shape[0]
            vars_db[variable]["end"] = counter

        for parameter in self.params_names:
            temp = tc.ocp._method.eval_at_control(
                tc.ocp, tc.parameters[parameter], 0
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
        print(vars_db)
        # setting the MPC function!
        # opti = tc.ocp.opti
        opti = tc.ocp._method.opti
        # self._mpc_fun = opti.to_function(
        #     "mpc_fun", [opti.p, opti.x, opti.lam_g], [opti.x, opti.lam_g, opti.f]
        # )
        self._mpc_fun = tc.ocp._method.opti.to_function(f_name, opti_xplam, opti_xplam)
        self._mpc_fun_cg = tc.ocp._method.opti.to_function(
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
                self._mpc_fun_cg.save(f_name + ".casadi")
            # self._mpc_fun = cs.external('f', './mpc_fun.so')

    ## obtain the solution of the ocp
    def _read_solveroutput(self, sol):

        sol_states = {}
        sol_controls = {}
        sol_variables = {}
        tc = self.tc

        if self._mpc_fun == None:
            for state in tc.states:
                _, sol_state = sol.sample(tc.states[state], grid="control")
                sol_states[state] = sol_state

            for control in tc.controls:
                _, sol_control = sol.sample(tc.controls[control], grid="control")
                sol_controls[control] = sol_control[0:-1]

            for variable in tc.variables:
                _, sol_variable = sol.sample(tc.variables[variable], grid="control")
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
                tc.ocp.set_initial(tc.states[state], sol_states[state].T)

            sol_controls = sol_ocp[1]
            for control in tc.controls:
                tc.ocp.set_initial(tc.controls[control], sol_controls[control].T)

            sol_variables = sol_ocp[2]
            for variable in tc.variables:
                tc.ocp.set_initial(tc.variables[variable], sol_variables[variable])

        # warm starting by shiting the solution by 1 step
        elif options == "shift":

            sol_states = sol_ocp[0]
            for state in tc.states:
                # print(state)
                # print(sol_states[state][1:].shape)
                # print(sol_states[state][-1:].shape)
                tc.ocp.set_initial(
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
                tc.ocp.set_initial(
                    tc.controls[control],
                    cs.vertcat(sol_controls[control][1:-1], zeros_np, zeros_np).T,
                )

            sol_variables = sol_ocp[2]
            for variable in tc.variables:
                tc.ocp.set_initial(tc.variables[variable], sol_variables[variable])

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

    # Internal function to apply the output of the MPC to the non-realtime bullet environment
    def _apply_control_nrbullet(self, sol_mpc, params_val):

        if self.parameters["control_type"] == "joint_velocity":
            control_info = self.parameters["control_info"]

            joint_indices = control_info["joint_indices"]
            if control_info["discretization"] == "constant_acceleration":
                # Computing the average of the first two velocities to apply as input
                # assuming constant acceleration input
                # control_action = (
                #     0.5 * (sol_mpc[0]["q_dot"][0, :] + sol_mpc[0]["q_dot"][1, :]).T
                # )

                for i in range(control_info["no_samples"]):
                    control_action = (
                        sol_mpc[0]["q_dot"][0, :] * (1 - i / control_info["no_samples"])
                        + sol_mpc[0]["q_dot"][1, :] * i / control_info["no_samples"]
                    ).T
                    self.world.setController(
                        control_info["robotID"],
                        "velocity",
                        joint_indices,
                        targetVelocities=control_action,
                    )
                    self.world.run_simulation(1)

                # simply giving the velocity at the next time step as the reference
                # control_action = sol_mpc[0]['q_dot'][1,:].T
                # future_joint_position = sol_mpc[0]['q'][0,:]
                # print("q_dot shape is ")
                # print(sol_mpc[0]['q_dot'].shape)
                # print("control action shape is ")
                # print(control_action.shape)
                # print("joint indices length is")
                # print(len(joint_indices))
            # self.world.setController(control_info['robotID'], 'velocity', joint_indices, targetPositions = future_joint_position, targetVelocities = control_action)
            if "force_control" in control_info:
                q_dot_force = control_info["fcon_fun"](
                    params_val["q0"], params_val["f_des"], params_val["f_meas"]
                )
                # print("qdot force is ")
                # print(q_dot_force)
                # cap the magnitude of q_dot_force
                # max_q_dot_force_norm = 0.1
                # norm_q_dot_force = cs.norm_1(q_dot_force)
                # if norm_q_dot_force > max_q_dot_force_norm:
                #    print("norm of q_dot_force:")
                #    print(norm_q_dot_force)
                #    q_dot_force = q_dot_force/norm_q_dot_force*max_q_dot_force_norm
                #    print("q dot force after normalization")
                #    print(q_dot_force)
                control_action = np.array(cs.vec(control_action) + q_dot_force)

            # control_action = control_action[0]
            # print(control_action)
            # print(q_dot_force)
            # self.world.setController(
            #     control_info["robotID"],
            #     "velocity",
            #     joint_indices,
            #     targetVelocities=control_action,
            # )
            # print("This ran")
            # print(sol_mpc[0]['s_dot'])
            # print(sol_mpc[0]['s'])

            # self.world.run_simulation(control_info["no_samples"])

        elif self.parameters["control_type"] == "joint_torque":

            control_info = self.parameters["control_info"]
            joint_indices = control_info["joint_indices"]
            control_action = sol_mpc[1]["tau"][0, :].T
            robot = self.parameters["params"]["robots"][control_info["robotID"]]

            for i in range(control_info["no_samples"]):
                self.world.setController(
                    control_info["robotID"],
                    "torque",
                    joint_indices,
                    targetTorques=control_action,
                )
                self.world.run_simulation(1)

        elif self.parameters["control_type"] == "joint_acceleration":
            # implemented using the inverse dynamics solver and applying joint torques

            control_info = self.parameters["control_info"]
            joint_indices = control_info["joint_indices"]
            control_action = sol_mpc[1]["q_ddot"][0, :].T
            robot = self.parameters["params"]["robots"][control_info["robotID"]]

            # compute joint torques through inverse dynamics of casadi model
            # joint_torques = np.array(robot.id(params_val['q0'], params_val['q_dot0'], control_action))

            # compute joint torques using inverse dynamics functions from pybullet
            joint_torques = self.world.computeInverseDynamics(
                control_info["robotID"],
                list(params_val["q0"]),
                list(params_val["q_dot0"]),
                list(control_action),
            )

            if "force_control" in control_info:
                # compute the feedforward torque needed to apply the desired EE force
                # print(control_info['jac_fun']([0]*7))
                torque_forces = cs.mtimes(
                    control_info["jac_fun"](params_val["q0"]).T, params_val["f_des"]
                )
                joint_torques = np.array(cs.vec(joint_torques) + cs.vec(torque_forces))

            # print(joint_torques)

            for i in range(control_info["no_samples"]):
                self.torque_effort_sumsqr += (
                    cs.sumsqr(joint_torques) * self.world.physics_ts
                )
                self.world.setController(
                    control_info["robotID"],
                    "torque",
                    joint_indices,
                    targetTorques=joint_torques,
                )
                self.world.run_simulation(1)

                if not "force_control" in control_info:
                    # computing the joint torque to apply at the same frequency as bullet simulation
                    params_innerloop = self._read_params_nrbullet()
                    joint_torques = self.world.computeInverseDynamics(
                        control_info["robotID"],
                        list(params_innerloop["q0"]),
                        list(params_innerloop["q_dot0"]),
                        list(control_action),
                    )

        elif self.parameters["control_type"] == "joint_position":

            print("Not implemented")

        else:

            raise Exception(
                "[Error] Unknown control type for bullet environment initialized"
            )

    # Internal function to read the values of the parameter variables from the bullet simulation environment
    # in non realtime case
    def _read_params_nrbullet(self):

        params_val = {}
        parameters = self.parameters

        for params_name in self.params_names:

            param_val = []
            param_info = parameters["params"][params_name]

            if param_info["type"] == "joint_position":

                jointsInfo = self.world.readJointState(
                    param_info["robotID"], param_info["joint_indices"]
                )
                for jointInfo in jointsInfo:
                    param_val.append(jointInfo[0])

                params_val[params_name] = param_val

            elif param_info["type"] == "joint_velocity":

                jointsInfo = self.world.readJointState(
                    param_info["robotID"], param_info["joint_indices"]
                )
                for jointInfo in jointsInfo:
                    param_val.append(jointInfo[1])

                params_val[params_name] = param_val

            elif param_info["type"] == "joint_torque":

                jointsInfo = self.world.readJointState(
                    param_info["robotID"], param_info["joint_indices"]
                )
                for jointInfo in jointsInfo:
                    param_val.append(jointInfo[3])

                params_val[params_name] = param_val

            elif param_info["type"] == "joint_force":

                jointsInfo = self.world.readJointState(
                    param_info["robotID"], param_info["joint_indices"]
                )
                forces = jointsInfo[0][2][0:3]
                print("Forces reading before correction")
                print(jointsInfo[0][2])
                forces_corrected = param_info["post_process"](
                    param_info["fk"], params_val["q0"], forces
                )

                params_val[params_name] = forces_corrected
                print("Force sensor readings:")
                print(forces_corrected)

            elif param_info["type"] == "progress_variable":
                if self.mpc_ran:
                    if "state" in param_info:

                        params_val[params_name] = self.sol_mpc[0][params_name[0:-1]][1]

                    elif "control" in param_info:

                        params_val[params_name] = self.sol_mpc[1][params_name[0:-1]][1]
                else:

                    params_val[params_name] = 0

            elif param_info["type"] == "set_value":

                params_val[params_name] = param_info["value"]

            elif param_info["type"] == "function_of_s":
                if self.mpc_ran:
                    _, normal = param_info["function"](self.sol_mpc[0]["s"][0])
                    params_val[params_name] = normal * param_info["gain"]
                else:
                    params_val[params_name] = np.array([0, 0, 0]).T
            else:

                print(
                    "[ERROR] Invalid type of parameter to be read from the simulation environment"
                )
        print(self.params_history.append(params_val))

        return params_val


# TODO: set a method to let the user define the inputs and outputs of the function get from opti.to_function
# TODO: This should also account for monitors
# TODO: Set offline solution for initialization ( mpc.set_offline_solution(solver, options, ?))

if __name__ == "__main__":

    print("No syntax errors")
