# Takes the task-specification and also the task context as an input and
# returns a COP
from sys import path

from rockit import (
    Ocp,
    DirectMethod,
    MultipleShooting,
    FreeTime,
    SingleShooting,
    DirectCollocation,
    UniformGrid,
    
)
import numpy as np
import casadi as cs
from tasho import input_resolution
from tasho import default_mpc_options
from tasho.utils import geometry

# This may be replaced by some method get_ocp_variables, which calls self.states, self.controls, ...
from collections import namedtuple

_OCPvars = namedtuple("OCPvars", ["q", "q_dot", "q_ddot", "q0", "q_dot0"])


class task_context:
    """Class for task context
    The class stores all expressions and constraints relevant to an OCP
    """

    def __init__(self, time=None, horizon_steps=10, name="tc", time_init_guess = 5):
        """Class constructor - initializes and sets the field variables of the class

        :param time: The prediction horizon of the OCP.
        :type time: float

        :param horizon_steps: Number of steps in the ocp horizon.
        :type horizon_steps: int

        :param name: Name of the task context
        :type name: string

        """

        ocp = Ocp()
        self.ocp = ocp
        self.horizon = []
        self.stages = []
        self.tc_name = name
        self.states = {}
        self.controls = {}
        self.variables = {}
        self.parameters = {}
        self.constraints = {}
        self.monitors = {}
        self.disc_settings = {}
        self.monitors_configured = False
        # self.opti = ocp.opti # Removed from Rockit
        self.t_ocp = None

        self.robots = {}
        self.OCPvars = None

        stage = self.create_stage(time, horizon_steps, time_init_guess = time_init_guess)

        self.tc_dict = {
            "states": {},
            "controls": {},
            "parameters": {},
            "variables": {},
            "inp_ports": [],
            "out_ports": [],
            "props": [],
        }

        # Setting Ipopt as the default ocp solver
        self.ocp_solver = "ipopt"
        self.ocp_options = {
            "ipopt": {"linear_solver": "mumps"},
            "error_on_fail": True,
        }
        self.set_ocp_solver(self.ocp_solver, self.ocp_options)

        # Setting sqpmethod as the default mpc solver
        def_dict = default_mpc_options.get_default_mpc_options()
        self.mpc_solver = "sqpmethod"
        self.mpc_options = def_dict["sqp_ip_mumps"]["options"]


    def create_stage(self, time = None, horizon_steps = 10, time_init_guess = 5):

        """
        Creates an OCP stage
        :param time: The duration of the prediction horizon. Free-time problem if no argument is provided.
        :type time: float

        :param horizon_steps: Number of steps in the prediction horizon.
        :type horizon_steps: int
        """
        if time == None:
            stage = self.ocp.stage(T = FreeTime(time_init_guess))
            self.ocp_rate = None
        else:
            stage = self.ocp.stage(T = time)
            self.ocp_rate = time / horizon_steps

        self.horizon.append(horizon_steps)
        self.stages.append(stage)

        # setting default discretization settings
        self.set_discretization_settings(self.disc_settings, stage = len(self.horizon) - 1)

        return stage

    def create_expression(self, name, type, shape, stage=0):

        """Creates a symbolic expression for variables in OCP.

        :param name: name of the symbolic variable
        :type name: string

        :param type: type of the symbolic variable. \n
                'state' - a variable that stands for a set of states that evolve over time as the states comprising the dynamical system of the OCP. \n
                'control' - For representing the control actions of the dynamical system of the OCP. \n
                'parameter' - Parameters of the dynamical system. Useful for representing quantities that might change over MPC iterations. eg: the initial conditions of the OCP. \n
                'variable' - A decision variable of the OCP that is not a state or the control action of the dynamical system.
        :type type: string

        :param shape: 2-dimensional tuple that denotes the dimensions of the expression.
        :type shape: tuple of int.

        """

        ocp = self.stages[stage]

        if type == "state":
            state = ocp.state(shape[0], shape[1])
            self.states[name] = [state, stage]

            return state

        elif type == "control":
            control = ocp.control(shape[0], shape[1])
            self.controls[name] = [control, stage]

            return control

        elif type == "parameter":
            parameter = ocp.parameter(shape[0], shape[1])
            self.parameters[name] = [parameter, stage]

            return parameter

        elif type == "variable":
            variable = ocp.variable(shape[0], shape[1])
            self.variables[name] = [variable, stage]

            return variable

        else:

            print("ERROR: expression type undefined")

    def create_state(
        self, name, shape=(1, 1), init_parameter=False, warm_start=1, stage=0
    ):
        """
        Creates a symbolic expression for state. If init_parameter is true, also
        creates a parameter corresponding to the initial condition of the state.
        Depending on init_parameter, returns state or state and parameter.

        :param name: name of the symbolic variable
        :type name: string

        :param shape: 2-dimensional tuple that denotes the dimensions of the expression.
        :type shape: tuple of int.

        :param init_parameter: Indicates whether an initial condition parameter should be created simultaneously.
        :type init_parameter: boolean

        :param warm_start: Indicates if and how the states should be warmstarted. 0 - no warm start. 1 - warm start with the initial value.
        :type warm_start: Int
        """
        ocp = self.stages[stage]
        if name in self.states:
            raise Exception("The state of the name " + name + " is already declared.")
        state = ocp.state(shape[0], shape[1])  # creating state
        self.states[name] = [state, stage]  # adding the symbolic variable to list of states
        self.tc_dict["states"][name] = {}  # recording state in the task context dict

        if init_parameter:
            parameter = self.create_parameter(name + "0", shape, stage = stage)
            ocp.subject_to(ocp.at_t0(state) == parameter)  # adding init eq constraint
            self.tc_dict["states"][name]["assoc_param"] = name + "0"  # associated param
            self.tc_dict["parameters"][name + "0"]["assoc_state"] = name
            self.tc_dict["inp_ports"][-1]["warm_start"] = warm_start
            self.tc_dict["inp_ports"][-1]["wvar"] = name
            return state, parameter

        return state

    def create_parameter(self, name, shape=(1, 1), port_or_property=1, stage=0, grid = None):
        """
        Creates a symbolic expression for a parameter. By default, also assigns a
        port which is relevant while deploying the controller on a robot.

        :param name: name of the symbolic variable
        :type name: string

        :param shape: 2-dimensional tuple that denotes the dimensions of the expression.
        :type shape: tuple of int.

        :param port_or_property: 1 - port is created. 2 - property is created. any other value - neither port nor property created
        :type port_or_property: int

        """
        ocp = self.stages[stage]
        if name in self.parameters:
            raise Exception(
                "The parameter of the name " + name + " is already declared."
            )

        if grid == None:
            parameter = ocp.parameter(shape[0], shape[1])
        else:
            parameter = ocp.parameter(shape[0], shape[1], grid = grid)
        self.parameters[name] = [parameter, stage]
        self.tc_dict["parameters"][name] = {}  # declaring param in the tc dict

        if port_or_property == 1:
            # declaring a port and making connections with the associated parameter
            self.tc_dict["inp_ports"].append(
                {
                    "name": "port_inp_" + name,
                    "var": name,
                    "desc": "[default] Read values for parameter " + name,
                }
            )
            self.tc_dict["parameters"][name]["assoc_port"] = (
                len(self.tc_dict["inp_ports"]) - 1
            )

        elif port_or_property == 2:
            # declaring a property
            self.tc_dict["props"].append(
                {
                    "name": "prop_inp_" + name,
                    "var": name,
                    "desc": "[default] Read values for parameter " + name,
                }
            )
            self.tc_dict["parameters"][name]["assoc_prop"] = (
                len(self.tc_dict["props"]) - 1
            )

        return parameter

    def create_control(self, name, shape=(1, 1), outport=True, stage=0):
        """
        Creates a symbolic expression for a control variable. By default, also
        assigns an output port which is relevant while deploying the controller
        on a robot.

        :param name: name of the symbolic variable
        :type name: string

        :param shape: 2-dimensional tuple that denotes the dimensions of the expression.
        :type shape: tuple of int.

        :param outport: Set to True if an output port needs to be created, False otherwise.

        """
        ocp = self.stages[stage]
        if name in self.controls:
            raise Exception(
                "The parameter of the name " + name + " is already declared."
            )
        control = ocp.control(shape[0], shape[1])
        self.controls[name] = [control, stage]
        self.tc_dict["controls"][name] = {}

        if outport == True:
            # declaring an port and making connections with the associated control
            self.tc_dict["out_ports"].append(
                {
                    "name": "port_out_" + name,
                    "var": name,
                    "desc": "[default] write values for control " + name,
                }
            )
            self.tc_dict["controls"][name]["assoc_port"] = (
                len(self.tc_dict["out_ports"]) - 1
            )

        return control

    def set_dynamics(self, state, state_der, stage=0):

        """Set dynamics of state variables of the OCP.

        :param state: expression of the state.
        :type state: state expression

        :param state_der: The derivative of state expression.
        :type state_der: const or another expression variable.

        """

        ocp = self.stages[stage]
        ocp.set_der(state, state_der)

    def add_regularization(
        self, expression, weight, norm="L2", variable_type="state", reference=0, stage=0
    ):

        """Add regularization to states or controls or variables. L1 regularization creates slack variables to avoid the non-smoothness in the optimization problem.

        :param expression: expression of the variable on which the regularization is to be added
        :type expression: casadi expression

        :param weight: The regularization weight
        :type weight: float or parameter

        :param norm: 'L2' for L2 regularization. 'L1' for L1 regularization.
        :type norm: String

        :param variable_type: 'state' for a state variable, 'control' for a control variable, 'variable' for a variable
        :type variable_type: String

        :param reference: The reference for regularization. 0 by default.
        :type reference: float or parameter
        """

        ocp = self.stages[stage]
        if norm == "L2":

            if variable_type == "state" or variable_type == "control":
                ocp.add_objective(
                    ocp.integral(cs.sumsqr(expression - reference), grid="control")
                    * weight
                )
            elif variable_type == "variable":
                ocp.add_objective(ocp.at_t0(cs.sumsqr(expression - reference)) * weight)
            else:
                raise Exception(
                    "invalid variable type. Must be 'state', 'control' or 'variable"
                )

        elif norm == "L1":

            if variable_type == "state" or variable_type == "control":
                slack_variable = self.create_expression(
                    "slack_path_con", "control", expression.shape
                )
                ocp.subject_to(
                    -slack_variable <= (expression - reference <= slack_variable)
                )
                ocp.add_objective(ocp.integral(cs.DM.ones(1, expression.shape[0])@slack_variable, grid="control") * weight)
            elif variable_type == "variable":
                slack_variable = self.create_expression(
                    "slack", "variable", expression.shape
                )
                ocp.subject_to(
                    -slack_variable <= (expression - reference <= slack_variable)
                )
                ocp.add_objective(ocp.at_t0(slack_variable) * weight)
            else:
                raise Exception(
                    "invalid variable type. Must be 'state', 'control' or 'variable"
                )

    def add_objective(self, obj, stage=0):

        """Add an objective function to the OCP

        :param obj: A casadi expression of the objective
        :type state: state expression

        """

        self.stages[stage].add_objective(obj)

    def add_task_constraint(self, task_spec, stage=0):

        ocp = self.stages[stage]

        if "initial_constraints" in task_spec:
            for init_con in task_spec["initial_constraints"]:
                if 'hard' in init_con and not init_con['hard']:
                    raise Exception("not implemented yet")

                else:
                    # hard initial constraints
                    if 'lub' in init_con:
                        ocp.subject_to(init_con['lower_limits'] <= (
                            ocp.at_t0(init_con["expression"]) <= init_con["upper_limits"])
                        )
                    else:
                        # assumed to be equality if not specified
                        ocp.subject_to(
                            ocp.at_t0(init_con["expression"]) == init_con["reference"],
                        )
                    

        if "final_constraints" in task_spec:
            for final_con in task_spec["final_constraints"]:

                if final_con["hard"]:

                    if "inequality" not in final_con and "lub" not in final_con:
                        ocp.subject_to(
                            ocp.at_tf(final_con["expression"]) == final_con["reference"]
                            )
                    elif "inequality" in final_con:
                        ocp.subject_to(
                            ocp.at_tf(final_con["expression"]) <= final_con["upper_limits"]
                            )
                    else:
                        ocp.subject_to(
                            final_con["lower_limits"] <= (ocp.at_tf(final_con["expression"]) <= final_con["upper_limits"])
                            )

                else:
                    if "norm" not in final_con or final_con["norm"] == "L2":
                        obj_con = (
                            ocp.at_tf(
                                cs.sumsqr(
                                    final_con["expression"] - final_con["reference"]
                                )
                            )
                            * final_con["gain"]
                        )
                        ocp.add_objective(obj_con)
                        if "name" in final_con:
                            self.constraints[final_con["name"]] = {"obj": obj_con}
                    elif final_con["norm"] == "L1":
                        slack_variable = self.create_expression(
                            "slack_variable", "variable", final_con["expression"].shape
                        )
                        ocp.subject_to(
                            -slack_variable
                            <= (
                                ocp.at_tf(
                                    final_con["expression"] - final_con["reference"]
                                )
                                <= slack_variable
                            )
                        )
                        obj_con = ocp.at_tf(
                            cs.DM.ones(final_con["expression"].shape).T @ slack_variable
                        )
                        ocp.add_objective(obj_con)
                        if "name" in final_con:
                            self.constraints[final_con["name"]] = {"obj": obj_con}

        if "path_constraints" in task_spec:
            for path_con in task_spec["path_constraints"]:

                if not "inequality" in path_con and not "lub" in path_con:
                    if not path_con["hard"]:

                        if "norm" not in path_con or path_con["norm"] == "L2":
                            # print('L2 norm added')
                            obj_con = (
                                ocp.integral(
                                    cs.sumsqr(
                                        path_con["expression"] - path_con["reference"]
                                    ),
                                    grid="control",
                                )
                                * path_con["gain"]
                            )
                            ocp.add_objective(obj_con)
                            if "name" in path_con:
                                self.constraints[path_con["name"]] = {"obj": obj_con}
                        elif path_con["norm"] == "L1":
                            # print("L1 norm added")
                            slack_variable = self.create_expression(
                                "slack_path_con",
                                "control",
                                path_con["expression"].shape,
                            )
                            ocp.subject_to(
                                -slack_variable
                                <= (
                                    path_con["reference"] - path_con["expression"]
                                    <= slack_variable
                                )
                            )
                            obj_con = (
                                ocp.integral(
                                    cs.DM.ones(path_con["expression"].shape).T
                                    @ slack_variable,
                                    grid="control",
                                )
                                * path_con["gain"]
                            )
                            ocp.add_objective(obj_con)
                            if "name" in path_con:
                                self.constraints[path_con["name"]] = {"obj": obj_con}
                        elif path_con["norm"] == "L2_nonsquared":
                            # print("L1 norm added")
                            slack_variable = self.create_expression(
                                "slack_path_con",
                                "control",
                                (1, 1),
                            )
                            ocp.subject_to(
                                cs.sumsqr(
                                    path_con["expression"] - path_con["reference"]
                                )
                                <= slack_variable ** 2
                            )
                            ocp.subject_to(slack_variable >= 0)
                            obj_con = (
                                ocp.integral(slack_variable, grid="control")
                                * path_con["gain"]
                            )
                            ocp.add_objective(obj_con)
                            # ocp.subject_to(slack_variable >= 0)

                            ocp.add_objective(obj_con)
                            if "name" in path_con:
                                self.constraints[path_con["name"]] = {"obj": obj_con}
                    elif path_con["hard"]:

                        ocp.subject_to(
                            path_con["expression"] == path_con["reference"],
                            include_first=False,
                        )

                elif "inequality" in path_con:

                    if path_con["hard"]:
                        if "include_first" in path_con:
                            ocp.subject_to(
                                path_con["expression"] <= path_con["upper_limits"]
                            )
                        else:
                            if "upper_limits" in path_con:
                                ocp.subject_to(
                                    path_con["expression"] <= path_con["upper_limits"],
                                    include_first=False,
                                )
                            elif "lower_limits" in path_con:
                                ocp.subject_to(
                                    path_con["expression"] >= path_con["lower_limits"],
                                    include_first=False,
                                )
                    else:
                        con_violation = cs.fmax(
                            path_con["expression"] - path_con["upper_limits"], 0
                        )
                        if "norm" not in path_con or path_con["norm"] == "L2":
                            obj = (
                                ocp.integral(con_violation ** 2, grid="control")
                                * path_con["gain"]
                            )
                            ocp.add_objective(obj)
                            if "name" in path_con:
                                self.constraints[path_con["name"]] = {"obj": obj}
                        elif path_con["norm"] == "L1":
                            slack_variable = self.create_expression(
                                "slack_path_con",
                                "control",
                                path_con["expression"].shape,
                            )
                            ocp.subject_to(
                                path_con["expression"] - path_con["upper_limits"]
                                <= slack_variable
                            )
                            ocp.subject_to(0 >= -slack_variable)
                            obj = (
                                ocp.integral(slack_variable, grid="control")
                                * path_con["gain"]
                            )
                            ocp.add_objective(obj)
                            if "name" in path_con:
                                self.constraints[path_con["name"]] = {"obj": obj}
                        elif path_con["norm"] == "squaredL2":
                            slack_variable = self.create_expression(
                                path_con["slack_name"], "control", (1, 1)
                            )
                            ocp.subject_to(
                                cs.sumsqr(path_con["expression"])
                                - path_con["upper_limits"]
                                <= slack_variable
                            )
                            ocp.subject_to(0 >= -slack_variable)
                            ocp.add_objective(
                                ocp.integral(slack_variable) * path_con["gain"]
                            )

                elif "lub" in path_con:

                    if path_con["hard"]:
                        if "include_first" in path_con:
                            ocp.subject_to(
                                (path_con["lower_limits"] <= path_con["expression"])
                                <= path_con["upper_limits"]
                            )
                        else:
                            ocp.subject_to(
                                (path_con["lower_limits"] <= path_con["expression"])
                                <= path_con["upper_limits"],
                                include_first=False,
                            )
                    else:
                        con_violation = cs.fmax(
                            path_con["expression"] - path_con["upper_limits"], 0
                        )
                        con_violation = con_violation + cs.fmax(
                            path_con["lower_limits"] - path_con["expression"], 0
                        )
                        if "norm" not in path_con or path_con["norm"] == "L2":
                            ocp.add_objective(
                                ocp.integral(con_violation) * path_con["gain"]
                            )
                        elif path_con["norm"] == "L1":
                            slack = ocp.control(path_con["expression"].shape[0])
                            ocp.subject_to(slack >= 0)
                            ocp.subject_to(
                                -slack + path_con["lower_limits"]
                                <= (
                                    path_con["expression"]
                                    <= path_con["upper_limits"] + slack
                                )
                            )
                            obj = (
                                ocp.integral(
                                    np.ones((1, path_con["expression"].shape[0]))
                                    @ slack,
                                    grid="control",
                                )
                                * path_con["gain"]
                            )
                            ocp.add_objective(obj)
                            if "name" in path_con:
                                self.constraints[path_con["name"]] = {"obj": obj}

                else:
                    raise Exception("ERROR: unknown type of path constraint added")
        if (
            "path_constraints" not in task_spec
            and "initial_constraints" not in task_spec
            and "final_constraints" not in task_spec
        ):
            raise Exception(
                "Unknown type of constraint added. Not 'path_constraints', 'final_constraints' or 'initial_constraints'"
            )

    def minimize_time(self, weight, stage=0):

        """Add a cost on minimizing the time of the OCP to provide time-optimal
        solutions.

        :param weight: A weight factor on cost penalizing the total time of the ocp horizon.
        :type weight: float
        """
        self.stages[stage].add_objective(weight * self.stages[stage].T)

    def set_ocp_solver(self, solver, options={}):

        """Choose the numerical solver for solving the OCP and set the options.

        :param solver: name of the solver. 'ipopt', 'sqpmethod'.
        :type solver: string

        :param options: Dictionary of options for the solver
        :type options: dictionary

        """

        ocp = self.ocp
        if "expand" not in options:
            options["expand"] = True
        ocp.solver(solver, options)

    def set_discretization_settings(self, settings, stage=0):

        """Set the discretization method of the OCP

        :param settings: A dictionary for setting the discretization method of the OCP with the fields and options given below. \n
                'discretization method'(string)- 'multiple_shooting' or 'single_shooting'. \n
                'order' (integer)- The order of integration. Minumum one. \n
                'integration' (string)- The numerical integration algorithm. 'rk' - Runge-Kutta4 method.
        :type settings: dictionary

        """

        ocp = self.stages[stage]
        if "discretization method" not in settings:
            disc_method = "multiple shooting"
        else:
            disc_method = settings["discretization method"]

        if "integration" not in settings:
            integrator = "rk"
        else:
            integrator = settings["integration"]

        if "order" not in settings:
            M = 1
        else:
            M = settings["order"]

        if disc_method == "multiple shooting":
            ocp.method(MultipleShooting(N=self.horizon[stage], M=M, intg=integrator))
        elif disc_method == "single shooting":
            ocp.method(SingleShooting(N=self.horizon[stage], M=M, intg=integrator))
        elif disc_method == "direct collocation":
            ocp.method(DirectCollocation(N=self.horizon[stage], M=M))
        else:
            raise Exception(
                "ERROR: discretization with "
                + settings["discretization method"]
                + " is not defined"
            )

    def solve_ocp(self):

        """solves the ocp and returns the rockit solution object"""

        ocp = self.ocp
        sol = ocp.solve()
        self.configure_monitors()  # The first solve of the ocp configures the monitors
        self.sol = sol
        return sol

    def set_value(self, expr, value, stage=0):

        ocp = self.stages[stage]
        ocp.set_value(expr, value)

    def sol_sample(self, expr, grid="control", stage=0,**kwargs):
        t, x_sol = self.sol(self.stages[stage]).sample(expr, grid=grid,**kwargs)
        return t, x_sol

    def sol_value(self, expr, stage=0):
        x_val = self.sol(self.stages[stage]).value(expr)
        return x_val

    def set_initial(self, expr, value, stage=0):
        self.stages[stage].set_initial(expr, value)

    # Add monitors to the task context
    def add_monitor(self, task_monitor):

        """Adds the monitor to the task context.

        :param task_monitor: A dictionary specifying the monitor
        :type task_monitor: monitor dictionary

        """

        self.monitors[task_monitor["name"]] = task_monitor
        self.monitors_configured = False

    # Configure the monitors
    def configure_monitors(self):

        """Configures all the monitors in the task context. Should be run only after the
        ocp solve is called atleast once."""
        if not self.monitors_configured:
            for monitor in self.monitors:
                self._configure_each_monitor(self.monitors[monitor])

        self.monitors_configured = True

    def function_primal_residual(self):

        """ Returns a function to compute the primal residual of the output of the solver"""

        opti = self.ocp._method.opti
        residual = cs.fmax(opti.g - opti.ubg, 0) + cs.fmax(-opti.g + opti.lbg, 0)
        residual_max = cs.mmax(residual)
        fun_pr = cs.Function("fun_pr", [opti.x, opti.p, opti.lam_g], [residual_max])

        return fun_pr

    # Internal function to configure each monitor
    def _configure_each_monitor(self, monitor, stage=0):

        opti = self.ocp._method.opti
        ocp = self.stages[stage]
        expr = monitor["expression"]
        # define the casadi function to compute the monitor value
        _, expr_sampled = ocp.sample(
            expr, grid="control"
        )  # the monitored expression over the ocp control grid
        # print(expr_sampled[0].shape)

        # enforcing that each monitor should be a scalar
        if expr_sampled[0].shape[0] != 1 and expr_sampled[0].shape[1] > 1:
            raise Exception("The monitor " + monitor["name"] + " is not a scalar")
        # print("Expr sampled is!!!!!!!!!!!!!!!!!!!")
        # print(expr_sampled[0])
        if "initial" in monitor:
            expr_fun = cs.Function(
                "monitor_" + monitor["name"],
                [opti.x, opti.p, opti.lam_g],
                [expr_sampled[0], opti.x],
            )
        elif "final" in monitor:
            expr_fun = cs.Function(
                "monitor_" + monitor["name"],
                [opti.x, opti.p, opti.lam_g],
                [expr_sampled[-1], opti.x],
            )
        elif "always" or "once" in monitor:
            expr_fun = cs.Function(
                "monitor_" + monitor["name"],
                [opti.x, opti.p, opti.lam_g],
                [expr_sampled, opti.x],
            )
        else:
            raise Exception(
                "Invalid setting: the timing of the monitor "
                + monitor["name"]
                + " not set"
            )

        monitor["expr_fun"] = expr_fun
        # define the python subfunction that would compute the truth value of the monitor
        # defining monitor function for lower condition on the expression
        if "lower" in monitor:
            if "initial" in monitor or "final" in monitor:

                def monitor_fun(
                    opti_xplam, expr_fun=expr_fun, reference=monitor["reference"]
                ):
                    return expr_fun(*opti_xplam)[0] <= reference

            elif "always" in monitor:

                def monitor_fun(
                    opti_xplam, expr_fun=expr_fun, reference=monitor["reference"]
                ):
                    truth_value = expr_fun(*opti_xplam)[0].T <= reference
                    return (sum(np.array(truth_value)) == truth_value.shape[0])[0]

            else:

                def monitor_fun(
                    opti_xplam, expr_fun=expr_fun, reference=monitor["reference"]
                ):
                    truth_value = expr_fun(*opti_xplam)[0].T <= reference
                    return (sum(np.array(truth_value)) != 0)[0]

        # defining monitor functions for greater condition on the expression
        elif "greater" in monitor:
            if "initial" in monitor or "final" in monitor:

                def monitor_fun(
                    opti_xplam, expr_fun=expr_fun, reference=monitor["reference"]
                ):
                    return expr_fun(*opti_xplam)[0] >= reference

            elif "always" in monitor:

                def monitor_fun(
                    opti_xplam, expr_fun=expr_fun, reference=monitor["reference"]
                ):
                    truth_value = expr_fun(*opti_xplam)[0].T >= reference
                    return (sum(truth_value) == truth_value.shape[0])[0]

            else:

                def monitor_fun(
                    opti_xplam, expr_fun=expr_fun, reference=monitor["reference"]
                ):
                    truth_value = expr_fun(*opti_xplam)[0].T >= reference
                    return (sum(truth_value) != 0)[0]

        # defining monitor functions for equal to condition on the expression
        elif "equal" in monitor:
            if "initial" in monitor or "final" in monitor in monitor:

                def monitor_fun(
                    opti_xplam,
                    expr_fun=expr_fun,
                    reference=monitor["reference"],
                    tolerance=monitor["tolerance"],
                ):
                    return np.fabs(expr_fun(*opti_xplam)[0] - reference) <= tolerance

            elif "always" in monitor:

                def monitor_fun(
                    opti_xplam,
                    expr_fun=expr_fun,
                    reference=monitor["reference"],
                    tolerance=monitor["tolerance"],
                ):
                    truth_value = (
                        np.fabs(expr_fun(*opti_xplam)[0] - reference).T <= tolerance
                    )
                    return (sum(truth_value) == truth_value.shape[0])[0]

            else:

                def monitor_fun(
                    opti_xplam,
                    expr_fun=expr_fun,
                    reference=monitor["reference"],
                    tolerance=monitor["tolerance"],
                ):
                    truth_value = (
                        np.fabs(expr_fun(*opti_xplam)[0] - reference).T <= tolerance
                    )
                    return (sum(truth_value) != 0)[0]

        else:
            raise Exception(
                "No valid comparison operator provided to the monitor "
                + monitor["name"]
            )

        # Assign the monitor function to the dictionary element of the the task context that defines the monitor
        monitor["monitor_fun"] = monitor_fun

    def generate_MPC_component(self, location, cg_opts):

        """
        Generates the OCP (and MPC) codes and the json file with all the information
        about the ocp and stores it in the desired location.

        :param location: The location where the generated components should be saved.
        :type location: string

        :param cg_opts: The code-generation options for the solvers.
        :type cg_opts: Python dictionary
        """

        # Generate the OCP component
        ocp_fun, vars_db = self.generate_controller(
            "_ocp", location, self.ocp_solver, self.ocp_options, cg_opts["ocp_cg_opts"]
        )
        self.ocp_fun = ocp_fun

        # generate the MPC component if required
        if "mpc" in cg_opts and cg_opts["mpc"]:
            mpc_fun, _ = self.generate_controller(
                "_mpc",
                location,
                self.mpc_solver,
                self.mpc_options,
                cg_opts["mpc_cg_opts"],
            )
            self.mpc_fun = mpc_fun
            # TODO: add monitor value in OCP and MPC xplm
            vars_db["mpc_fun_name"] = self.tc_name + "_mpc"
            vars_db["mpc_file"] = location + self.tc_name + "_mpc.casadi"
            # TODO: also add the case where .c files are generated

        # Updating the json file and dumping it
        vars_db["num_inp_ports"] = len(self.tc_dict["inp_ports"])
        vars_db["num_out_ports"] = len(self.tc_dict["out_ports"])
        vars_db["num_props"] = len(self.tc_dict["props"])
        vars_db["num_states"] = len(self.states)
        vars_db["num_controls"] = len(self.controls)
        vars_db["num_parameters"] = len(self.parameters)
        vars_db["states"] = list(self.states.keys())
        vars_db["controls"] = list(self.controls.keys())
        vars_db["inp_ports"] = self.tc_dict["inp_ports"]
        vars_db["out_ports"] = self.tc_dict["out_ports"]
        vars_db["props"] = self.tc_dict["props"]
        vars_db["horizon"] = self.horizon[0] #TODO: what to do here really?
        vars_db["mpc_ts"] = self.ocp_rate
        vars_db["ocp_fun_name"] = self.tc_name + "_ocp"
        vars_db["ocp_file"] = location + self.tc_name + "_ocp.casadi"

        return vars_db

    def generate_controller(self, name, location, solver, sol_opts, cg_opts):

        """
        Generates code/.casadi files for the OCP.

        :param name: Name of the code-generated function.
        :type name: String

        :param location: The directory where the generated controller files should be saved.
        :type location: String.

        :param solver: Name of the solver
        :type solver: String.

        :param sol_opts: Options for the solver.
        :type sol_opts: Python dictionary

        :param cg_opts: Code-generation options
        :type opts: Python dictionary
        """

        # generate the ocp function
        # set the ocp solver
        self.set_ocp_solver(solver, sol_opts)
        self.ocp.solver(solver, sol_opts)
        # set the discretization settings and transcribe
        # self.set_discretization_settings(self.disc_settings)
        # self.stages[0]._method.main_transcribe(self.stages[0])
        ocp_xplm, vars_db, ocp_xplm_out = self._unroll_controller_vars()
        if not cg_opts["jit"]:
            ocp_fun = self.ocp.to_function(self.tc_name + name, ocp_xplm, ocp_xplm_out)
        else:
            jit_opts = {
                "jit": True,
                "compiler": "shell",
                "jit_options": {
                    "verbose": True,
                    "compiler": "ccache gcc",
                    "compiler_flags": self.parameters["codegen"]["flags"],
                },
                "verbose": False,
                "jit_serialize": "embed",
            }
            ocp_fun = self.ocp.to_function(
                self.tc_name + name, ocp_xplm, ocp_xplm_out, jit_opts
            )

        if cg_opts["save"]:
            ocp_fun.save(location + self.tc_name + name + ".casadi")

        if cg_opts["codegen"]:
            ocp_fun.generate(
                location + self.tc_name + name + ".c", {"with_header": True}
            )

        return ocp_fun, vars_db

    def _unroll_controller_vars(self):
        # unrolls all the variables in the task context into a single large vector
        # and also stores the meta data concerning the variables in a dictionary

        op_xplm = []  # declaring the vector roll
        vars_db = {}
        counter = 0

        for state in self.states.keys():
            stage = self.states[state][1]
            ocp = self.stages[stage]
            _, temp = ocp.sample(self.states[state][0], grid="control")
            temp2 = []
            for i in range(self.horizon[stage] + 1):
                temp2.append(temp[:, i])
            vars_db[state] = {"start": counter, "size": temp.shape[0]}
            op_xplm.append(cs.vcat(temp2))
            counter += temp.shape[0] * temp.shape[1]
            vars_db[state]["end"] = counter

        # obtain the opti variables related to control
        for control in self.controls.keys():
            stage = self.controls[control][1]
            ocp = self.stages[self.controls[control][1]]
            _, temp = ocp.sample(self.controls[control][0], grid="control")
            temp2 = []
            for i in range(self.horizon[stage]):
                temp2.append(
                    ocp._method.eval_at_control(ocp, self.controls[control][0], i)
                )
            vars_db[control] = {
                "start": counter,
                "size": temp.shape[0],
                "jump": temp.shape[0],
            }
            counter += temp.shape[0] * (temp.shape[1] - 1)
            vars_db[control]["end"] = counter
            op_xplm.append(cs.vcat(temp2))

        # obtain the opti variables related for variables
        for variable in self.variables.keys():
            stage = self.variables[variable][1]
            ocp = self.stages[stage]
            temp = ocp._method.eval_at_control(ocp, self.variables[variable][0], 0)
            op_xplm.append(temp)
            vars_db[variable] = {
                "start": counter,
                "size": temp.shape[0],
                "jump": temp.shape[0],
            }
            counter += temp.shape[0]
            vars_db[variable]["end"] = counter

        for parameter in self.parameters.keys():
            stage = self.parameters[parameter][1]
            ocp = self.stages[stage]
            temp = ocp._method.eval_at_control(ocp, self.parameters[parameter][0], 0)
            op_xplm.append(temp)
            vars_db[parameter] = {
                "start": counter,
                "size": temp.shape[0],
                "jump": temp.shape[0],
            }
            counter += temp.shape[0]
            vars_db[parameter]["end"] = counter

        ocp = self.stages[0]
        op_xplm.append(self.ocp._method.opti.lam_g)
        vars_db["lam_g"] = {
            "start": counter,
            "size": self.ocp._method.opti.lam_g.shape[0],
        }
        counter += self.ocp._method.opti.lam_g.shape[0]
        vars_db["lam_g"]["end"] = counter


        import copy

        op_xplm_out = copy.deepcopy(op_xplm)

        #Appending the time_grid to the output
        for st in range(len(self.stages)):
            stage_now = self.stages[st]
            _, t_grid = stage_now.sample(stage_now.t, grid = "control")
            op_xplm_out.append(t_grid.T)
            vars_db['time_grid_stage_'+str(st)] = {}
            vars_db['time_grid_stage_'+str(st)]['start'] = counter
            counter += t_grid.shape[1]
            vars_db['time_grid_stage_'+str(st)]['end'] = counter
            vars_db['time_grid_stage_'+str(st)]['size'] = t_grid.shape[1]

        # Add the monitor function to op_xplm
        monitors = self.monitors
        monitors_dict = {}
        for m_name in monitors.keys():

            # Compute the value of the monitor function at the desired location
            _, expr_mon_grid = ocp.sample(
                monitors[m_name]["expression"], grid="control"
            )
            if "initial" in monitors[m_name]:
                expr_m = expr_mon_grid[0]
                print(expr_m)
            elif "final" in monitors[m_name]:
                expr_m = expr_mon_grid[-1]
            elif "always" in monitors[m_name]:
                expr_m = cs.mmax(expr_mon_grid)
            elif "any" in monitors[m_name]:
                expr_m = cs.mmin(expr_mon_grid)
            else:
                raise Exception("Monitor not set at initial, final, always or any")

            # Add the monitor function to op_xplm and store the name and location
            op_xplm_out.append(expr_m)
            monitors_dict[m_name] = counter
            counter += 1

        # Add an array of the monitor names and the dictionary associated with the locations
        # to vars_db
        monitor_names = list(monitors_dict.keys())
        vars_db["monitor_names"] = monitor_names
        vars_db["monitor_locations"] = monitors_dict

        if "termination_criteria" not in monitor_names:
            raise Warning("No termination criteria set as a monitor!")

        return [cs.vcat(op_xplm)], vars_db, [cs.vcat(op_xplm_out)]

    def set_input_resolution(self, robot):

        if robot.input_resolution == "velocity":

            raise Exception("ERROR: Not implemented and probably not recommended")

        elif robot.input_resolution == "acceleration":

            q, q_dot, q_ddot, q0, q_dot0 = input_resolution.acceleration_resolved(
                self, robot, {}
            )

            self.OCPvars = _OCPvars(q, q_dot, q_ddot, q0, q_dot0)

        elif robot.input_resolution == "torque":

            raise Exception("ERROR: Not implemented")
            # q, q_dot, q_ddot, tau, q0, q_dot0 = input_resolution.torque_resolved(
            #     self, robot, {"forward_dynamics_constraints": False}
            # )

            # self.OCPvars = _OCPvars(q, q_dot, tau, q0, q_dot0)

        else:

            raise Exception(
                'ERROR: Only available options for input_resolution are: "velocity", "acceleration" or "torque".'
            )

    def add_robot(self, robot):
        self.robots[robot.name] = robot
        # robot.transcribe(self)
        self.set_input_resolution(robot)

        # self.sim_system_dyn = robot.sim_system_dyn(self.task_context)

    @property
    def get_states(self):
        states = cs.vertcat()
        for st in self.states.values():
            states = cs.vertcat(states, st)
        return states

    @property
    def get_controls(self):
        controls = cs.vertcat()
        for co in self.controls.values():
            controls = cs.vertcat(controls, co)
        return controls

    @property
    def get_parameters(self):
        parameters = cs.vertcat()
        for par in self.parameters.values():
            parameters = cs.vertcat(parameters, par)
        return parameters

    def get_output_states(self):
        states = self.get_states
        return self.ocp.sample(states, grid="control")[1]
        # return cs.vertcat(self.ocp.sample(self.get_states, grid='integrator', refine=refine)[1], self.ocp.sample(self.get_controls, grid='integrator', refine=refine)[1],)

    #
    def get_output_controls(self):
        controls = self.get_controls
        return self.ocp.sample(controls, grid="control")[1]
