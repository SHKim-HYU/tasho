import sys
from tasho import task_prototype_rockit as tp
from tasho import input_resolution, WorldSimulator, MPC
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
import time
from rockit import MultipleShooting, Ocp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':


    robot = rob.Robot('atlas')
    position = [0.0, 0.0, 0.9]
    orientation = [0.0, 0.0, 0.0, 1.0]
    obj = WorldSimulator.WorldSimulator(plane_spawn = True, bullet_gui = True)
    obj.visualization_realtime = True
    time.sleep(2.0)
    atlasID = obj.add_robot(position, orientation, 'atlas', fixedBase = False)
    obj.run_simulation(20)
    time.sleep(20.0)
