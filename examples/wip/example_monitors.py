# An example file demonstrating the usage of monitors
# shall simulate a mass-spring-damper and put a monitor
# on it's mechanical energy. There would be no control action


from tasho import task_prototype_rockit as tp

horizon_size = 10
t_mpc = 0.1

tc = tp.task_context(horizon_size*t_mpc)

x = tc.create_expression('x', 'state', (1,1))
x_dot = tc.create_expression('x_dot', 'state', (1,1))
x0 = tc.create_expression('x0', 'parameter', (1,1))
xdot_0 = tc.create_expression('xdot_0', 'parameter', (1,1))

tc.set_dynamics(x, x_dot)
tc.set_dynamics(x_dot, -1*x_dot - 1*x) #mass, viscous co-efficient and the spring constant are all 1

x_initcon = {'expression':x, 'reference':x0}
x_dot_initcon = {'expression':x_dot, 'reference':xdot_0}
tc.add_task_constraint({'initial_constraints':[x_initcon, x_dot_initcon]})
tc.ocp.set_value(x0, 1)
tc.ocp.set_value(xdot_0, 1)
disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
tc.set_discretization_settings(disc_settings)
tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3, "print_level":0}, "print_time":False})

#adding the monitors now
ke = 0.5*x**2 + 0.5*x_dot**2
tc.add_monitor({"name":"first", "expression":ke, "reference":0.1, "lower":True, "initial":True}) #check if KE is lower than 0.1 at the first step of OCP
tc.add_monitor({"name":"final", "expression":ke, "reference":0.1, "greater":True, "final":True})#check if KE is lower than 0.1 at the final step of OCP
tc.add_monitor({"name":"once", "expression":ke, "reference":0.1, "lower":True, "once":True})#check if KE is lower than 0.1 at the final step of OCP
tc.add_monitor({"name":"always", "expression":ke, "reference":0.1, "lower":True, "always":True})#check if KE is lower than 0.1 at the final step of OCP

sol = tc.solve_ocp() #to configure the monitors

#create opti function
# TODO: Remove opti dependencie
opti = tc.opti


for i in range(30):

	print("Iteration number:" + str(i+1))
	#sample all the opti variables
	_, optix = sol.sample(opti.x, grid = 'control')
	optix = optix[0]
	_, optip = sol.sample(opti.p, grid = 'control')
	optip = optip[0]
	_, optilam_g = sol.sample(opti.lam_g, grid = 'control')
	optilam_g = optilam_g[0]

	#compute the truth value of the monitors and print
	print(tc.monitors["first"]["monitor_fun"]([optix, optip, optilam_g]))
	print(tc.monitors["final"]["monitor_fun"]([optix, optip, optilam_g]))
	print(tc.monitors["once"]["monitor_fun"]([optix, optip, optilam_g]))
	print(tc.monitors["always"]["monitor_fun"]([optix, optip, optilam_g]))

	_, x_sol = sol.sample(x, grid= "control")
	_, xdot_sol = sol.sample(x_dot, grid= "control")

	print("position = " + str(x_sol[0]) + " and velocity = " + str(xdot_sol[0]))

	tc.ocp.set_value(x0, x_sol[1])
	tc.ocp.set_value(xdot_0, xdot_sol[1])

	sol = tc.solve_ocp()
