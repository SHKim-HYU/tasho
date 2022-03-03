# This shows a very simple application of the multi-stage application using Tasho

from tkinter import HORIZONTAL
from tasho.task_prototype_rockit import task_context as tp
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from tasho.utils import dist_computation
import time

def bounce_stage(stage):

    # tc = task_context(horizon_size = 20)
    x = tc.create_state('x'+str(stage), (1,1), stage = stage)
    xd = tc.create_expression('xd'+str(stage), 'variable', (1,1), stage = stage)

    y = tc.create_state('y'+str(stage), (1,1), stage = stage)
    yd = tc.create_state('yd'+str(stage), (1,1), stage = stage)
    ydd = -9.81

    tc.set_dynamics(x, xd, stage = stage)
    tc.set_dynamics(y, yd, stage = stage)
    tc.set_dynamics(yd, ydd, stage = stage)

    y_con = {"hard":True, 'inequality':True, 'expression':-y, 'upper_limits':0}
    y_hit_ground = {"hard":True, 'expression':y, 'reference':0}
    tc.add_task_constraint({"path_constraints":[y_con]}, stage = stage)
    tc.add_task_constraint({"final_constraints":[y_hit_ground]}, stage = stage)

    tc.add_regularization(x, 1e-3, stage=stage)
    tc.add_regularization(y, 1e-3, stage=stage)
    tc.add_regularization(yd, 1e-3, stage=stage)



    return [x, xd, y, yd]

horizon_steps = 20
coeff = 0.9 #coefficient of restitution

tc = tp(horizon_steps=horizon_steps, time_init_guess=1.0)
stage0 = tc.stages[0]
state0 = bounce_stage(0)

stage1 = tc.create_stage(horizon_steps=horizon_steps, time_init_guess=1.0)
state1 = bounce_stage(1)

# Add inter-stage constraints
for i in range(3):
    tc.ocp.subject_to(stage0.at_tf(state0[i]) == stage1.at_t0(state1[i]))

# collision with ground impulse
tc.ocp.subject_to(stage0.at_tf(state0[3])*0.9 == -stage1.at_t0(state1[3]))
tc.ocp.subject_to(stage0.at_tf(state0[1]) == stage1.at_t0(state1[1]))

init_vel = {'hard':True, 'lub':True, 'expression':cs.vertcat(state0[1], state0[3]), 'lower_limits':[-10, -10], 'upper_limits':[10,10]}
init_pos = {'hard':True, 'expression':cs.vertcat(state0[0], state0[2]), 'reference': [0, 1]}
tc.add_task_constraint({'initial_constraints':[init_pos, init_vel]}, stage = 0)

tc.set_initial(state1[2], 1, stage = 1)
tc.set_initial(state0[2], 1, stage = 0)
# tc.ocp.set_initial(stage0.T, 1)

final_pos = {'hard':True, 'equality':True, 'expression':state1[0], 'reference':10}
tc.add_task_constraint({'final_constraints':[final_pos]}, stage = 1)

tc.solve_ocp()


_, x_traj0 = tc.sol_sample(state0[0], stage = 0)
_, x_traj1 = tc.sol_sample(state1[0], stage = 1)

_, y_traj0 = tc.sol_sample(state0[2], stage = 0)
_, y_traj1 = tc.sol_sample(state1[2], stage = 1)

plt.figure()
plt.plot(x_traj0, y_traj0)
plt.plot(x_traj1, y_traj1)
plt.show()