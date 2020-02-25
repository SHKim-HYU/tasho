import sys
from tasho import task_prototype_rockit as tp
from rockit import MultipleShooting, Ocp
import matplotlib.pyplot as plt

#Constructs a simple OCP to verify that the basic functions of task_prototype_rockit is in order

if __name__ == '__main__':
    
    tc = tp.task_context(5)

    x = tc.create_expression('x', 'state', (1, 1))

    u = tc.create_expression('u', 'control', (1, 1))

    tc.set_dynamics(x, u)

    task_spec = {'initial_constraints':[{'expression':x, 'reference':0}]}
    task_spec['path_constraints'] = [{'expression':u, 'reference':0, 'hard':False, 'gain':1}]
    task_spec['final_constraints'] = [{'expression':x, 'reference':5, 'hard':True}]

    tc.add_task_constraint(task_spec)

    tc.ocp.solver('ipopt')
    tc.ocp.method(MultipleShooting(N = 5, M = 2, intg='rk'))

    ocp = tc.ocp
    sol = ocp.solve()
    t, x_val= sol.sample(x, grid='control')


    plt.figure()
    plt.plot(t, x_val)
    plt.show(block=True)