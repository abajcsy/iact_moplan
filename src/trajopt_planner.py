import numpy as np
from numpy import linalg
from numpy import linspace
import matplotlib.pyplot as plt
import time
import math
import json

from sympy import symbols
from sympy import lambdify

import trajoptpy
import or_trajopt
import openravepy
from openravepy import *

import openrave_utils
from openrave_utils import *

import logging
import copy

OBS_CENTER = [-1.3858/2.0 - 0.1, -0.1, 0.0]
HUMAN_CENTER = [0.0, 0.2, 0.0]

class TrajoptPlanner(Planner):
	"""
	This class plans a trajectory from start to goal 
	with TrajOpt. 
	"""

	def __init__(self):

		# ---- important internal variables ---- #
		Planner.__init__(self)

		self.weights = 1
		self.waypts_prev = None

		# ---- OpenRAVE Initialization ---- #
		
		# initialize robot and empty environment
		model_filename = 'jaco_dynamics'
		self.env, self.robot = initialize(model_filename)

		# insert any objects you want into environment
		self.bodies = []
	
		# plot the table and table mount
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		plotCabinet(self.env)

	# ---- custom feature and cost functions ---- #

	def featurize(self, waypts):
		"""
		Computes the user-defined features for a given trajectory.
		---
		input trajectory, output list of feature values
		"""
		features = [None,None]
		features[0] = self.velocity_features(waypts)
		features[1] = [0.0]*(len(waypts)-1)
		for index in range(0,len(waypts)-1):
			features[1][index] = self.table_features(waypts[index+1])
		return features

	# -- Velocity -- #

	def velocity_features(self, waypts):
		"""
		Computes total velocity cost over waypoints, confirmed to match trajopt.
		---
		input trajectory, output scalar feature
		"""
		vel = 0.0
		for i in range(1,len(waypts)):
			curr = waypts[i]
			prev = waypts[i-1]
			vel += np.linalg.norm(curr - prev)**2
		return vel
	
	def velocity_cost(self, waypts):
		"""
		Computes the total velocity cost.
		---
		input trajectory, output scalar cost
		"""
		#mywaypts = np.reshape(waypts,(7,self.num_waypts_plan)).T
		return self.velocity_features(mywaypts)

	# -- Distance to Table -- #

	def table_features(self, waypt):
		"""
		Computes the total cost over waypoints based on 
		z-axis distance to table
		---
		input trajectory, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EEcoord_z = coords[6][2]
		return EEcoord_z
	
	def table_cost(self, waypt):
		"""
		Computes the total distance to table cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.table_features(waypt)
		return feature*self.weights

	# ---- custom constraints --- #

	def table_constraint(self, waypt):
		"""
		Constrains z-axis of robot's end-effector to always be 
		above the table.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[10]
		EE_coord_z = EE_link.GetTransform()[2][3]
		if EE_coord_z > 0:
			EE_coord_z = 0
		return -EE_coord_z

	# ---- here's trajOpt --- #
		
	def trajOpt(self, start, goal):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input is start and goal pos, updates the waypts_plan
		"""
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]), 1)
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4
		if self.waypts_plan == None:
			init_waypts = np.zeros((self.num_waypts_plan,7))
			for count in range(self.num_waypts_plan):
				init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)
		else:
			init_waypts = self.waypts_plan 
		
		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"max_iter" : 40
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [1.0]}
			}
			],
			"constraints": [
			{
				"type": "joint",
				"params": {"vals": goal.tolist()}
			}
			],
			"init_info": {
                "type": "given_traj",
                "data": init_waypts.tolist()
			}
		}

		s = json.dumps(request)
		prob = trajoptpy.ConstructProblem(s, self.env)

		for t in range(1,self.num_waypts_plan): 
			prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)

		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

	# ---- replanning ---- #

	def replan(self, start, goal, start_time, final_time, step_time, weights):
		"""
		Replan the trajectory from start to goal given weights.
		---
		input trajectory parameters, update raw and upsampled trajectories
		"""
		if weights == None:
			return
		self.start = start
		self.goal = goal
		self.start_time = start_time
		self.final_time = final_time
		self.curr_waypt_idx = 0
		self.weights = weights
		self.trajOpt(start, goal)
		self.upsample(step_time)

		# return the upsampled waypts
		return self.waypts

	def kill_planner(self):
		"""
		Destroys openrave thread and environment for clean shutdown
		"""
		self.env.Destroy()
		RaveDestroy() # destroy the runtime

