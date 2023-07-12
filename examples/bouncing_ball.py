# This shows a simple example for multi-stage optimization using Tasho

from tasho import TaskModel
from tasho.ConstraintExpression import ConstraintExpression
from examples.templates.Regularization import Regularization
from tasho.OCPGenerator import OCPGenerator
import casadi as cs
import numpy as np


# Defining a toy task, for a ball that is falling freely in space
def ball_task(name):
    bouncing_ball = TaskModel.Task(name, "2d_freefall")

    x = bouncing_ball.create_variable('x', 'pos', 'state', (1,1))
    y = bouncing_ball.create_variable('y', 'pos', 'state', (1,1))

    xd = bouncing_ball.create_variable('x', 'vel', 'variable', (1,1))
    yd = bouncing_ball.create_variable('y', 'vel', 'state', (1,1))
    grav = bouncing_ball.create_variable('grav', 'constant', 'magic_number', (1,1), -9.81)

    bouncing_ball.set_der(x, xd)
    bouncing_ball.set_der(y, yd)
    bouncing_ball.set_der(yd, grav)

    y_above_ground = ConstraintExpression("ball_above_ground", "", y, 'hard', lb = 0)
    y_hit_ground = ConstraintExpression("ball_hit_ground", "eq", y, 'hard', reference = 0)

    # adding regularization for numerical reasons, will not affect the dynamics
    bouncing_ball.add_path_constraints(y_above_ground, 
                                    Regularization(x, 1e-3), 
                                    Regularization(y, 1e-3),
                                    Regularization(yd, 1e-3))
    bouncing_ball.add_terminal_constraints(y_hit_ground)

    return bouncing_ball

horizon_steps = 20
coeff = 0.85 #coefficient of restitution

number_bounces = 5
start_pos = [0,1]
goal_x = 10

bounce_tasks = []
for i in range(number_bounces):
    bounce_tasks.append(ball_task("bounce_" + str(i)))

# add initial pose constraints
bounce_tasks[0].add_initial_constraints(
    ConstraintExpression("x", "start_pos", bounce_tasks[0].variables['pos_x'], 'hard', reference = start_pos[0]),
    ConstraintExpression("y", "start_pos", bounce_tasks[0].variables['pos_y'], 'hard', reference = start_pos[1]))

# add position constraint at the end of the bounce`
bounce_tasks[-1].add_terminal_constraints(
    ConstraintExpression("x", "final_pos_con", bounce_tasks[-1].variables['pos_x'], 'hard', reference = goal_x))

transcription_options = {"horizon_steps" : horizon_steps}
OCP_gen = OCPGenerator(bounce_tasks[0], True, transcription_options)

# append the next bounces to create multi-stage problem
for i in range(1, number_bounces):
    collision_impulse = [lambda a, b: coeff*a+b, [bounce_tasks[i-1].variables['vel_y']], [bounce_tasks[i].variables['vel_y']], 0]
    x_vel_const = [lambda a, b: a - b, [bounce_tasks[i-1].variables['vel_x']], [bounce_tasks[i-1].variables['vel_x']], 0]
    OCP_gen.append_task(bounce_tasks[i], True, transcription_options, exclude_continuity=[bounce_tasks[i].variables["vel_y"]], generic_inter_stage_constraints= [collision_impulse, x_vel_const])

tc = OCP_gen.tc
OCP_gen.tc.solve_ocp()

x_traj_log = []
y_traj_log = []
ocpx_sampler = []
ocpy_sampler = []
ocp_t_limit = []
SIMULATION = True
ANIMATE = True
if SIMULATION:
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(number_bounces):


        t_grid, x_traj = tc.sol_sample(OCP_gen.stage_tasks[i].variables['pos_x'].x, stage = i)
        ocpx_sampler.append(tc.sol(tc.stages[i]).sampler(OCP_gen.stage_tasks[i].variables['pos_x'].x))
        ocpy_sampler.append(tc.sol(tc.stages[i]).sampler(OCP_gen.stage_tasks[i].variables['pos_y'].x))
        ocp_t_limit.append(t_grid[-1])
        _, y_traj = tc.sol_sample(OCP_gen.stage_tasks[i].variables['pos_y'].x, stage = i)
        

        x_traj_log = x_traj_log + list(x_traj)
        y_traj_log = y_traj_log + list(y_traj)

        plt.plot(x_traj, y_traj)

    plt.show()
    
    # compute equisampled data for realistic visualization
    x_traj_log = []
    y_traj_log = []
    frame_rate = 1/45
    for i in range(number_bounces):
        t_range = np.arange(0, ocp_t_limit[i],step = frame_rate )
        x_traj_log = cs.vertcat(x_traj_log, ocpx_sampler[i](t_range))
        y_traj_log = cs.vertcat(y_traj_log, ocpy_sampler[i](t_range))


    data = (cs.vertcat(cs.DM(x_traj_log).T, cs.DM(y_traj_log).T)).full()
    fig = plt.figure()
    plt.plot([-50, 50], [0, 0]) # plot the ground
    plt.plot([goal_x], [0], 'gx')
    ball, = plt.plot([], [], 'ro', lw = 50)
    plt.xlim(-2, 12)
    plt.ylim(-0.2, 1.2)

    if ANIMATE:

        import matplotlib.animation as animation
        time_template = 'time = %.1fs'
        # time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        # def init():
        #     ball.set_data([], [])
        #     time_text.set_text('')
        #     return ball, time_text

        def update_line(num, data, line):
            line.set_data([data[0,num]], [data[1,num]])
            return line,

        ani = animation.FuncAnimation(fig, update_line, data.shape[1], fargs= (data, ball),
                              interval=25, blit=True)
        ani.save('im.mp4')
        plt.show()

        
