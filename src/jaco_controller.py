#! /usr/bin/env python
"""
This codes implements ROS and OpenRAVE trajectory tracking for the Jaco2 7DOF robot.

Author: Andrea Bajcsy (abajcsy@berkeley.edu)
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import pid
import tf
import sys, select, os
import thread
import argparse
import actionlib
import time
import ros_utils
import phri_planner

import openravepy
from openravepy import *

import openrave_utils
from openrave_utils import *

import kinova_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
from kinova_msgs.srv import *
from std_msgs.msg import Float32
from sympy import Point, Line

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

prefix = 'j2s7s300_driver'

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]

epsilon = 0.10
MAX_CMD_TORQUE = 40.0
INTERACTION_TORQUE_THRESHOLD = 8.0

class JacoController(object): 
	"""
	This class represents a node that moves the Jaco with PID control.
	The joint velocities are computed as:
		
		V = -K_p(e) - K_d(e_dot) - K_i*Integral(e)

	where:

		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K_p = accounts for present values of position error
		K_i = accounts for past values of error, accumulates error over time
		K_d = accounts for possible future trends of error, based on current rate of change
	
	Subscribes to: 
		/j2s7s300_driver/out/joint_angles	- Jaco sensed joint angles
		/j2s7s300_driver/out/joint_torques	- Jaco sensed joint torques
	
	Publishes to:
		/j2s7s300_driver/in/joint_velocity	- Jaco commanded joint velocities 
	
	Required parameters:
		sim_flag 				  - flag for if in simulation or not
		admittance_flag 	- flag for if the robot is compliant (T) or stiff (F)
		planner 					- Planner object that contains trajectory to execute
	"""

	def __init__(self, planner, sim_flag=True, admittance_flag=False):
		"""
		Setup of the ROS node. Publishing computed torques happens at 100Hz.
		"""

		self.sim_flag = sim_flag
		self.admittance_flag = admittance_flag 

		# ---- Trajectory Setup ---- #

		# store the planner object
		self.planner = planner

		# save intermediate target position from degrees (default) to radians 
		self.target_pos = self.planner.start.reshape((7,1))
		# save start configuration of arm
		self.start_pos = self.planner.start.reshape((7,1))
		# save final goal configuration
		self.goal_pos = self.planner.goal.reshape((7,1))

		# track if you have gotten to start/goal of path
		self.reached_start = False
		self.reached_goal = False

		# keeps running time since beginning of program execution
		self.process_start_T = time.time() 
		# keeps running time since beginning of path
		self.path_start_T = None 

		# ----- Controller Setup ----- #

		# stores maximum COMMANDED joint torques		
		self.max_cmd = MAX_CMD_TORQUE*np.eye(7)
		# stores current COMMANDED joint torques
		self.cmd = np.eye(7) 
		# stores current joint MEASURED joint torques
		self.joint_torques = np.zeros((7,1))

		# P, I, D gains 
		p_gain = 50.0
		i_gain = 0.0
		d_gain = 20.0
		self.P = p_gain*np.eye(7)
		self.I = i_gain*np.eye(7)
		self.D = d_gain*np.eye(7)
		self.controller = pid.PID(self.P,self.I,self.D,0,0)

		if self.sim_flag:
			self.start_OPENRAVE()
		else:
			self.start_ROS()

	def start_ROS(self):
		"""
		Creates the ROS node, publishers, and subscribers for the real robot.
		"""
		if self.admittance_flag:
			# start admittance control mode
			self.start_admittance_mode()

		# ---- ROS Setup ---- #

		rospy.init_node("jaco_controller")

		# create joint-velocity publisher
		self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# create subscriber to joint_torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)

		# publish to ROS at 100hz
		r = rospy.Rate(100) 

		print "----------------------------------"
		print "Moving robot, press ENTER to quit:"
		
		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
			r.sleep()
		
		print "----------------------------------"

		if self.admittance_flag:
			# end admittance control mode
			self.stop_admittance_mode()

	def start_OPENRAVE(self):
		"""
		Creates an OPENRAVE instance to simulate robot moving through trajectory.
		"""

		# initialize robot and empty environment
		model_filename = 'jaco_dynamics'
		self.env, self.robot = initialize(model_filename)

		# insert any objects you want into environment
		self.bodies = []
	
		# plot the table and table mount
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		plotCabinet(self.env)

		# simulate robot moving through the planned waypts
		for curr_pos in self.planner.waypts:
			pos = np.append(curr_pos.reshape(7), np.array([0,0,0]), 1)
			pos[2] += math.pi
			self.robot.SetDOFValues(pos)
			time.sleep(self.planner.step_time)


	def start_admittance_mode(self):
		"""
		Switches Jaco to admittance-control mode using ROS services
		"""
		service_address = prefix+'/in/start_force_control'	
		rospy.wait_for_service(service_address)
		try:
			startForceControl = rospy.ServiceProxy(service_address, Start)
			startForceControl()           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def stop_admittance_mode(self):
		"""
		Switches Jaco to position-control mode using ROS services
		"""
		service_address = prefix+'/in/stop_force_control'	
		rospy.wait_for_service(service_address)
		try:
			stopForceControl = rospy.ServiceProxy(service_address, Stop)
			stopForceControl()           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def PID_control(self, pos):
		"""
		Return a control torque based on PID control
		"""
		error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
		return -self.controller.update_PID(error)

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot 
		"""
		# read the current joint torques from the robot
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		planner_type = type(self.planner)

		# if using the phri_planner, then learn from measured joint torques
		if planner_type is phri_planner.PHRIPlanner:
			interaction = False
			for i in range(7):
				THRESHOLD = INTERACTION_TORQUE_THRESHOLD
				if self.reached_start and i == 3:
					THRESHOLD = 2.0
				if self.reached_start and i > 3:
					THRESHOLD = 2.0
				if np.fabs(torque_curr[i][0]) > THRESHOLD:
					interaction = True
				else:
					# zero out torques below threshold for cleanliness
					torque_curr[i][0] = 0.0

			# if experienced large enough interaction force, then deform traj
			if interaction:
				print "--- INTERACTION ---"
				print "u_h: " + str(torque_curr)

				self.planner.learn_weights(torque_curr)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""
		# read the current joint angles from the robot
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		curr_pos = curr_pos*(math.pi/180.0)	

		# update the OpenRAVE simulation 
		#self.planner.update_curr_pos(curr_pos)

		# update target position to move to depending on:
		# - if moving to START of desired trajectory or 
		# - if moving ALONG desired trajectory
		self.update_target_pos(curr_pos)

		# update cmd from PID based on current position
		self.cmd = self.PID_control(curr_pos)

		# check if each angular torque is within set limits
		for i in range(7):
			if self.cmd[i][i] > self.max_cmd[i][i]:
				self.cmd[i][i] = self.max_cmd[i][i]
			if self.cmd[i][i] < -self.max_cmd[i][i]:
				self.cmd[i][i] = -self.max_cmd[i][i]

	def update_target_pos(self, curr_pos):
		"""
		Takes the current position of the robot. Determines what the next
		target position to move to should be depending on:
		- if robot is moving to start of desired trajectory or 
		- if robot is moving along the desired trajectory 
		"""
		# check if the arm is at the start of the path to execute
		if not self.reached_start:

			dist_from_start = -((curr_pos - self.start_pos + math.pi)%(2*math.pi) - math.pi)			
			dist_from_start = np.fabs(dist_from_start)

			# check if every joint is close enough to start configuration
			close_to_start = [dist_from_start[i] < epsilon for i in range(7)]

			# if all joints are close enough, robot is at start
			is_at_start = all(close_to_start)

			if is_at_start:
				self.reached_start = True
				self.path_start_T = time.time()
			else:
				#print "NOT AT START"
				# if not at start of trajectory yet, set starting position 
				# of the trajectory as the current target position
				self.target_pos = self.start_pos
		else:
			#print "REACHED START --> EXECUTING PATH"

			t = time.time() - self.path_start_T
			#print "t: " + str(t)

			# get next target position from position along trajectory
			self.target_pos = self.planner.interpolate(t)

			# check if the arm reached the goal, and restart path
			if not self.reached_goal:
			
				dist_from_goal = -((curr_pos - self.goal_pos + math.pi)%(2*math.pi) - math.pi)			
				dist_from_goal = np.fabs(dist_from_goal)

				# check if every joint is close enough to goal configuration
				close_to_goal = [dist_from_goal[i] < epsilon for i in range(7)]
			
				# if all joints are close enough, robot is at goal
				is_at_goal = all(close_to_goal)
			
				if is_at_goal:
					self.reached_goal = True
			else:
				print "REACHED GOAL! Holding position at goal."
				self.target_pos = self.goal_pos

"""
if __name__ == '__main__':

	if len(sys.argv) < 4:
		print "ERROR: Not enough arguments. Specify sim_flag, admittance_flag, final_time"
	else:	
		sim_flag = True if sys.argv[1] == "T" or sys.argv[1] == "t" else sim_flag = False
		admittance_flag = True if sys.argv[2] == "T" or sys.argv[2] == "t" else admittance_flag = False
		final_time = float(sys.argv[3])

	JacoController(sim_flag, admittance_flag, final_time)
"""
