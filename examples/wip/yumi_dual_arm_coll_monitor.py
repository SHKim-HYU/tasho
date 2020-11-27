# Monitor to compute the distance between Yumi arms and also which arm is at the centre

# An example file demonstrating the usage of monitors
# shall simulate a mass-spring-damper and put a monitor 
# on it's mechanical energy. There would be no control action


from tasho import task_prototype_rockit as tp
import numpy as np
from tasho import robot as rob
import casadi as cs

tc = tp.task_context(1.0)

q = tc.create_expression('q', 'state', (18,1))
rob_settings = {'n_dof' : 18, 'no_links' : 20, 'q_min' : np.array([-2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, -0.1, -0.1, -2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, -0.1, -0.1]).T, 'q_max' : np.array([2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025, 2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025]).T }
tc.set_dynamics(q, np.array([0]*18))
robot = rob.Robot('yumi')

#computing the forward kinematics of the robot tree
fk_vals = robot.fk(q)
fk_left_ee = fk_vals[7]
fk_right_ee = fk_vals[17]
disc_settings = {'discretization method': 'multiple shooting', 'horizon size': 1, 'order':1, 'integration':'rk'}
tc.set_discretization_settings(disc_settings)
tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3, "print_level":0}, "print_time":False})

#adding the monitors now
tc.add_monitor({"name":"workspace_left_free", "expression":fk_left_ee[1,3], "reference":0.10, "greater":True, "initial":True}) #check if KE is lower than 0.1 at the first step of OCP
tc.add_monitor({"name":"workspace_right_free", "expression":fk_right_ee[1,3], "reference":-0.10, "lower":True, "initial":True}) #check if KE is lower than 0.1 at the first step of OCP

tc.ocp.set_value(q, 0)
sol = tc.solve_ocp() #to configure the monitors

# opti = tc.ocp._method.opti
# _, optix = sol.sample(opti.x, grid = 'control')
# optix = optix[0]
# _, optip = sol.sample(opti.p, grid = 'control')
# optip = optip[0]
# _, optilam_g = sol.sample(opti.lam_g, grid = 'control')
# optilam_g = optilam_g[0]

# print(optix)
# print(optip)
# print(optilam_g)

# print(tc.monitors["workspace_right_free"]["monitor_fun"]([optix, optip, optilam_g]))
# print(tc.monitors["workspace_left_free"]["monitor_fun"]([optix, optip, optilam_g]))

q_sym = cs.SX.sym('q_sym', 18, 1)
monitorexpr_wlf = tc.monitors["workspace_left_free"]["monitor_fun"]([cs.vertcat(q_sym, np.array([0]*18)), [], np.array([0]*18)])
monitorexpr_wrf = tc.monitors["workspace_right_free"]["monitor_fun"]([cs.vertcat(q_sym, np.array([0]*18)), [], np.array([0]*18)])

mon_fun_wlf = cs.Function('mon_fun_wlf', [q_sym], [monitorexpr_wlf])
mon_fun_wrf = cs.Function('mon_fun_wrf', [q_sym], [monitorexpr_wrf])
joint_pos = np.array([-0.117, -0.52, 1.29, 0.53, -0.19, 0.08, -1.88, 0.0, 0.0, 0.574, -0.554, -1.55, 0.07, 2.63, -0.43, -0.45, 0.0, 0.0])
print(mon_fun_wlf(joint_pos))
print(mon_fun_wrf(joint_pos))

mon_fun_wlf.save("/home/ajay/Desktop/casadi_libaries_from_tasho/multirob_ee_2020/mon_fun_wlf.casadi")
mon_fun_wrf.save("/home/ajay/Desktop/casadi_libaries_from_tasho/multirob_ee_2020/mon_fun_wrf.casadi")
