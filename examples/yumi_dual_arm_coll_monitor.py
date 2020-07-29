# Monitor to compute the distance between Yumi arms and also which arm is at the centre

# An example file demonstrating the usage of monitors
# shall simulate a mass-spring-damper and put a monitor 
# on it's mechanical energy. There would be no control action


from tasho import task_prototype_rockit as tp


tc = tp.task_context(1.0)

q = tc.create_expression('q', 'variable', (18,1))
rob_settings = {'n_dof' : 18, 'no_links' : 20, 'q_min' : np.array([-2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, -0.1, -0.1, -2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, -0.1, -0.1]).T, 'q_max' : np.array([2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025, 2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025]).T }

robot = rob.Robot('yumi')

#computing the forward kinematics of the robot tree
fk_vals = robot.fk(q)
fk_left_ee = fk_vals[7]
fk_right_ee = fk_vals[17]
disc_settings = {'discretization method': 'multiple shooting', 'horizon size': 1, 'order':1, 'integration':'rk'}
tc.set_discretization_settings(disc_settings)
tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3, "print_level":0}, "print_time":False})

#adding the monitors now
tc.add_monitor({"name":"workspace_left_free", "expression":fk_left_ee[1,3], "reference":0.15, "greater":True, "initial":True}) #check if KE is lower than 0.1 at the first step of OCP
tc.add_monitor({"name":"workspace_right_free", "expression":fk_right_ee[1,3], "reference":-0.15, "lower":True, "initial":True}) #check if KE is lower than 0.1 at the first step of OCP

sol = tc.solve_ocp() #to configure the monitors

#create opti function 
# opti = tc.opti


# for i in range(30):

# 	print("Iteration number:" + str(i+1))
# 	#sample all the opti variables
# 	_, optix = sol.sample(opti.x, grid = 'control')
# 	optix = optix[0]
# 	_, optip = sol.sample(opti.p, grid = 'control')
# 	optip = optip[0]
# 	_, optilam_g = sol.sample(opti.lam_g, grid = 'control')
# 	optilam_g = optilam_g[0]

# 	#compute the truth value of the monitors and print
# 	print(tc.monitors["first"]["monitor_fun"]([optix, optip, optilam_g]))
# 	print(tc.monitors["final"]["monitor_fun"]([optix, optip, optilam_g]))
# 	print(tc.monitors["once"]["monitor_fun"]([optix, optip, optilam_g]))
# 	print(tc.monitors["always"]["monitor_fun"]([optix, optip, optilam_g]))

# 	_, x_sol = sol.sample(x, grid= "control")
# 	_, xdot_sol = sol.sample(x_dot, grid= "control")

# 	print("position = " + str(x_sol[0]) + " and velocity = " + str(xdot_sol[0]))

# 	tc.ocp.set_value(x0, x_sol[1])
# 	tc.ocp.set_value(xdot_0, xdot_sol[1])

# 	sol = tc.solve_ocp()
