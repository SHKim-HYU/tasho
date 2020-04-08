#Simulator file, running either a physics engine or a simple integrator model for velocity control of a robot
#This file is meant to be used for verifying the performance of algorithms in simulation.
#The class should have robot related variables that are updated continiously in run_simulation for communication
#with a middleware (ROS)

import numpy as np
import pybullet as p
import pybullet_data
import time

class world_simulator:

	def __init__(self):

		self.verbose = True
		self.physics_ts = 1.0/240.0 #Time step for the bullet environment for physics simulation
		physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
		p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
		p.setGravity(0,0,-9.81)
		planeId = p.loadURDF("plane.urdf")
		self.robotIDs = []
		self.objectIDs = []

	## add_cylinder(radius, height, weight, pose)
	# Function to add a uniform cylinder. Closely follows the interface of pybullet
	def add_cylinder(self, radius, height, weight, pose):

		if self.verbose:
			print("Adding a cylinder to the bullet environment")

		position = pose['position']
		orientation = pose['orientation']
		collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius = radius, height = height)
		visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius = radius, rgbaColor=[0, 0, 0, 1], specularColor=[0.4, .4, 0], length = height )
		cylinderID = p.createMultiBody(1, collisionShapeId, visualShapeId, position,orientation)

		self.objectIDs = cylinderID
		return cylinderID

	def add_robot(self, position, orientaion, robot_name = None, robot_urdf = None, fixedBase = True):

		if robot_name != None:
			print("Not implemented")
		elif robot_urdf != None:
			robotID = p.loadURDF(robot_urdf, position, orientation, useFixedBase=fixedBase)
		else:
			print("[Error] No valid robot name or robot urdf given")
			return

		self.robotIDs.append(robotID)
		return robotID

	def add_object_urdf(self, position, orientaion, object_urdf, fixedBase = False):

		objectID = p.loadURDF(object_urdf, position, orientation, useFixedBase=fixedBase)
		self.objectIDs.append(objectID)
		return objectID

	def readJointState(self):

		print("Not implemented")

	def setController(self, controller_type):

		if controller_type == 'velocity':
			print("Not implemented")
		elif controller_type == 'position':
			print("not implemented")
		elif controller_type == 'torque':
			print("Not implemented")
		else:
			print("unknown controller type")

	## run_simulation(N = 4)
	# Function to run the simulation in the bullet world. By default, simulates upto ~16.6 ms
	# @params N The number of timesteps of the physics time step (1/240 by default in bullet) per function call 
	def run_simulation(self, N = 4):

		for i in range(N):

			p.stepSimulation()
			time.sleep(self.physics_ts)

	## end_simulation()
	# Ends the simulation, disconnects the bullet environment.
	def end_simulation(self):

		if self.verbose:

			print("Ending simulation")
			p.disconnect()


if __name__ == '__main__':
	obj = world_simulator()

	position = [0.5, 0.0, 0.25]
	orientation = [0.0, 0.0, 0.0, 1.0]
	pose = {'position':position, 'orientation':orientation}
	name = 'cylinder1'
	height = 0.5
	radius = 0.2
	weight = 100

	obj.add_cylinder(radius, height, weight, pose)

	obj.run_simulation(480)

	obj.end_simulation()


