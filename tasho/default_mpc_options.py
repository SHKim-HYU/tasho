from copy import deepcopy


def get_default_mpc_options():
    """ Returns default mpc options that are stored in a large dictionary for re-use"""

    op_dict = {}

    # ipopt with lbfgs
    op_dict["ipopt_lbfgs"] = {}
    ip_lbfgs = op_dict["ipopt_lbfgs"]  # short form
    ip_lbfgs["solver_name"] = "ipopt"
    ip_lbfgs["options"] = {
        "ipopt": {
            "max_iter": 1000,
            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 5,
            "tol": 1e-3,
        },
        "error_on_fail": True,
    }

    op_dict["ipopt_lbfgs_hsl"] = deepcopy(op_dict["ipopt_lbfgs"])
    op_dict["ipopt_lbfgs_hsl"]["options"]["ipopt"]["linear_solver"] = "ma27"

    # exact hessian ipopt
    op_dict["ipopt"] = {"solver_name": "ipopt"}
    op_dict["ipopt"]["options"] = {
        "ipopt": {
            "max_iter": 25,
            "tol": 1e-3,
            "mu_init": 1e-3,
            "linear_solver": "mumps",
            "fixed_variable_treatment": "make_parameter",
            "hessian_constant": "no",
            "jac_c_constant": "no",
            "jac_d_constant": "no",
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
            "ma27_ignore_singularity": "yes",
        },
        "error_on_fail": True,
    }

    op_dict["ipopt_hsl"] = deepcopy(op_dict["ipopt"])
    op_dict["ipopt_hsl"]["options"]["ipopt"]["linear_solver"] = "ma27"

    # Now setting up the default options for the SQPMethods
    kkt_tol_pr = 1e-3
    kkt_tol_du = 1e-1
    min_step_size = 1e-4
    max_iter = 5

    # QRQP method
    max_iter_ls = 3
    op_dict["sqp_qrqp"] = {"solver_name": "sqpmethod"}
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
    op_dict["sqp_qrqp"]["options"] = solver_options

    # OSQP _method
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
    op_dict["sqp_osqp"] = {"solver_name": "sqpmethod", "options": solver_options}

    # QPOASES
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
    }
    op_dict["sqp_qpoases"] = {"solver_name": "sqpmethod", "options": solver_options}

    # SQP ipopt mumps
    max_iter = 5
    ipopt_max_iter = 20
    max_iter_ls = 0
    ipopt_tol = 1e-3
    tiny_step_tol = 1e-6
    mu_init = 0.001
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
        "linear_solver": "mumps",
        "mumps_pivtolmax": 1e-6,
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
        "ma27_ignore_singularity": "yes",
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
    }
    op_dict["sqp_ip_mumps"] = {"solver_name": "sqpmethod", "options": solver_options}

    # sqp ipopt hsl
    op_dict["sqp_ip_hsl"] = deepcopy(op_dict["sqp_ip_mumps"])
    op_dict["sqp_ip_hsl"]["options"]["qpsol_options"]["nlpsol_options"]["ipopt"][
        "linear_solver"
    ] = "ma27"

    return op_dict


if __name__ == "__main__":
    import pprint

    mpc_def_options = get_default_mpc_options()
    pprint.pprint(mpc_def_options)
