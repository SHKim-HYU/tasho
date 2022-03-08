# This shows a simple example for multi-stage optimization using Tasho

from tasho import TaskModel
from tasho.ConstraintExpression import ConstraintExpression
from examples.templates.Regularization import Regularization
from tasho.OCPGenerator import OCPGenerator


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
coeff = 0.9 #coefficient of restitution

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

SIMULATION = False
if SIMULATION:
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(number_bounces):


        _, x_traj = tc.sol_sample(OCP_gen.stage_tasks[i].variables['pos_x'].x, stage = i)
        _, y_traj = tc.sol_sample(OCP_gen.stage_tasks[i].variables['pos_y'].x, stage = i)

        plt.plot(x_traj, y_traj)

    plt.show()