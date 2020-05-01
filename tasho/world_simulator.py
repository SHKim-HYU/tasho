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

		# Add button for
		p.addUserDebugParameter("Disconnect",1,0,1)

		# Set default camera position
		p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=55, cameraPitch=-10, cameraTargetPosition=[0,0,0.5])

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

	def add_robot(self, position, orientation, robot_name = None, robot_urdf = None, fixedBase = True):

		if robot_name != None:
			if robot_name == 'yumi':
				robotID = p.loadURDF("models/robots/ABB/yumi/yumi.urdf", position, orientation, useFixedBase=fixedBase)
			elif robot_name == 'kinova':
				robotID = p.loadURDF("models/robots/Kinova/Gen3/kortex_description/urdf/JACO3_URDF_V11.urdf", position, orientation, useFixedBase=fixedBase)
			elif robot_name == 'panda':
				print("not implemented")
			elif robot_name == 'iiwa7':
				robotID = p.loadURDF("kuka_iiwa/model.urdf", position, orientation, useFixedBase=fixedBase)
			else:
				print("[Error] No valid robot for the given robot name")
		elif robot_urdf != None:
			robotID = p.loadURDF(robot_urdf, position, orientation, useFixedBase=fixedBase)
		else:
			print("[Error] No valid robot name or robot urdf given")
			return

		self.robotIDs.append(robotID)
		# p.setVRCameraState(robotID)

		return robotID

	def add_object_urdf(self, position, orientaion, object_urdf, fixedBase = False):

		objectID = p.loadURDF(object_urdf, position, orientation, useFixedBase=fixedBase)
		self.objectIDs.append(objectID)
		return objectID

	def getJointInfoArray(self, robotID):

		num_joints = p.getNumJoints(robotID)

		joint_info_array = []
		for i in range(num_joints):
			joint_info_array.append(p.getJointInfo(robotID, i))

		return joint_info_array

	def readJointState(self, robotID, joint_indices):

		jointStates = p.getJointStates(robotID, joint_indices)

		return jointStates


	def resetJointState(self, robotID, joint_indices, new_joint_states):

		for i in range(len(joint_indices)):
			p.resetJointState(robotID, joint_indices[i], new_joint_states[i])

		return True


	def setController(self, robotID, controller_type, joint_indices, targetPositions = [None, None], targetVelocities = None, targetForces = None):

		if controller_type == 'velocity':
			if targetPositions[0] != None:
				p.setJointMotorControlArray(robotID, joint_indices, p.VELOCITY_CONTROL, targetPositions = targetPositions, targetVelocities = targetVelocities)
			else:
				p.setJointMotorControlArray(robotID, joint_indices, p.VELOCITY_CONTROL, targetVelocities = targetVelocities)
		elif controller_type == 'position':
			p.setJointMotorControlArray(robotID, joint_indices, p.POSITION_CONTROL, targetPositions = targetPositions, targetVelocities = targetVelocities)
		elif controller_type == 'torque':
			p.setJointMotorControlArray(robotID, joint_indices, p.TORQUE_CONTROL, targetForces = targetForces)
		else:
			print("unknown controller type")

	## run_simulation(N = 4)
	# Function to run the simulation in the bullet world. By default, simulates upto ~16.6 ms
	# @params N The number of timesteps of the physics time step (1/240 by default in bullet) per function call
	def run_simulation(self, N = 4):

		for i in range(N):

			p.stepSimulation()
			time.sleep(self.physics_ts)

	def run_continouous_simulation(self):
		run_sim = True
		# q_key = ord('q')
		enter_key = p.B3G_RETURN
		while (run_sim):
			keys = p.getKeyboardEvents()
			for k, v in keys.items():
				if (k == enter_key and (v & p.KEY_WAS_TRIGGERED)):
					run_sim = False
					print("Pressed Enter. Exit")

			p.stepSimulation()
			time.sleep(self.physics_ts)

	## end_simulation()
	# Ends the simulation, disconnects the bullet environment.
	def end_simulation(self):

		if self.verbose:

			print("Ending simulation")
			p.disconnect()

	## Computes the separating hyperplane for each link of the given robot and the given set of joint poses
	def compute_sh_bullet(self, robotID, q, distance = 0.1):

		print("Not implemented")

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

	#adding yumi robot to the bullet environment
	position = [-0.5, 0.0, 0.25]
	yumiID = obj.add_robot(position, orientation, 'yumi')

	jointInfo = obj.getJointInfoArray(yumiID)
	print(jointInfo)
	obj.run_simulation(480)

	obj.end_simulation()
