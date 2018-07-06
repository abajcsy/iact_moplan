#! /usr/bin/env python
"""
This code implements a trajectory-optimization based planner that updates
the weights on the cost function based on physical human-robot interaction (PHRI).

The underlying optimization is based on "Finding Locally Optimal, Collision-Free Trajectories 
with Sequential Convex Optimization" by Schulman, J. et al. 2013. 

The weight update is based on "Learning Learning Robot Objectives from Physical 
Human Robot Interaction" by Bajcsy, A. et. al. 2017.

Author: Andrea Bajcsy (abajcsy@berkeley.edu)
"""
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

import planner
import openrave_utils
from openrave_utils import *

import logging
import copy

OBS_CENTER = [-1.3858/2.0 - 0.1, -0.1, 0.0]
HUMAN_CENTER = [0.0, 0.2, 0.0]

FEATURE_OPTIONS = ["table", "coffee", "human", "laptop", "origin"]

UPDATE_GAINS = {"table":2.0, "coffee":2.0, "human":100.0, "laptop":100.0, "origin":100.0}
MAX_WEIGHTS = {"table":1.0, "coffee":1.0, "human":10.0, "laptop":10.0, "origin":10.0} 


class PHRIPlanner(planner.Planner):
	"""
	This class plans a trajectory from start to goal 
	with TrajOpt and updates feature weights from physical interaction.
	"""

	def __init__(self, features):

		# ---- important internal variables ---- #
		super(PHRIPlanner, self).__init__()

		self.weights = [None]*len(features)
		self.waypts_prev = None

		for feat in features:
			if feat not in FEATURE_OPTIONS:
				raise ValueError(feat + " is not a valid feature.")
				
		# list of features to consider
		self.features = features

		# ---- OpenRAVE Initialization ---- #
		
		# initialize robot and empty environment
		model_filename = 'jaco_dynamics'
		self.env, self.robot = initialize(model_filename, visualize=False)

		# insert any objects you want into environment
		self.bodies = []
	
		# plot the table and table mount
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		plotCabinet(self.env)
		if "laptop" in self.features:
			plotLaptop(self.env,self.bodies)
		if "origin" in self.features:
			plotSphere(self.env,self.bodies,OBS_CENTER,0.4)
		if "human" in self.features:
			plotSphere(self.env,self.bodies,HUMAN_CENTER,0.4)

		# ---- DEFORMATION Initialization ---- #

		self.alpha = -0.01
		self.n = 5
		self.A = np.zeros((self.n+2, self.n)) 
		np.fill_diagonal(self.A, 1)
		for i in range(self.n):
			self.A[i+1][i] = -2
			self.A[i+2][i] = 1
		self.R = np.dot(self.A.T, self.A)
		Rinv = np.linalg.inv(self.R)
		Uh = np.zeros((self.n, 1))
		Uh[0] = 1
		self.H = np.dot(Rinv,Uh)*(np.sqrt(self.n)/np.linalg.norm(np.dot(Rinv,Uh)))

	# ---- custom feature and cost functions ---- #

	def featurize(self, waypts):
		"""
		Computes the user-defined features for a given trajectory.
		---
		input trajectory, output list of feature values
		"""
		# always include velocity feature + custom features
		curr_features = [None]*(len(self.features)+1)
		curr_features[0] = self.velocity_features(waypts)
		idx = 1
		for feat in self.features:
			curr_features[idx] = [0.0]*(len(waypts)-1)
			for waypt_idx in range(0,len(waypts)-1):
				if feat is "table":
					curr_features[idx][waypt_idx] = self.table_features(waypts[waypt_idx+1])
				if feat is "coffee":
					curr_features[idx][waypt_idx] = self.coffee_features(waypts[waypt_idx+1])
				if feat is "human":
					curr_features[idx][waypt_idx] = self.human_features(waypts[waypt_idx+1], waypts[waypt_idx])
				if feat is "laptop":
					curr_features[idx][waypt_idx] = self.laptop_features(waypts[waypt_idx+1], waypts[waypt_idx])
				if feat is "origin":
					curr_features[idx][waypt_idx] = self.origin_features(waypts[waypt_idx+1])
			idx += 1

		return curr_features

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

	# -- Distance to Robot Base (origin of world) -- #

	def origin_features(self, waypt):
		"""
		Computes the total cost over waypoints based on 
		y-axis distance to table
		---
		input trajectory, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EEcoord_y = coords[6][1]
		EEcoord_y = np.linalg.norm(coords[6])
		print "EEcoord_y: " + str(EEcoord_y)
		return EEcoord_y
	
	def origin_cost(self, waypt):
		"""
		Computes the total distance from EE to base of robot cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.origin_features(waypt)
		weight_idx = self.features.index("origin")
		return feature*self.weights[weight_idx]										

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
		weight_idx = self.features.index("table")
		return feature*self.weights[weight_idx]

	# -- Coffee (or z-orientation of end-effector) -- #

	def coffee_features(self, waypt):
		"""
		Computes the distance to table cost for waypoint
		by checking if the EE is oriented vertically.
		Note: [0,0,1] in the first *column* corresponds to the cup upright
		---
		input trajectory, output scalar cost
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return sum(abs(EE_link.GetTransform()[:2,:3].dot([1,0,0])))
		
	def coffee_cost(self, waypt):
		"""
		Computes the total coffee (EE orientation) cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.coffee_features(waypt)
		weight_idx = self.features.index("coffee")
		return feature*self.weights[weight_idx]

	# -- Distance to Laptop -- #

	def laptop_features(self, waypt, prev_waypt):
		"""
		Computes laptop cost over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions
		---
		input trajectory, output scalar feature
		"""
		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.laptop_dist(inter_waypt)
		return feature

	def laptop_dist(self, waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where 
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		laptop_xy = np.array(OBS_CENTER[0:2])
		dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	def laptop_cost(self, waypt):
		"""
		Computes the total distance to laptop cost
		---
		input trajectory, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.laptop_features(curr_waypt,prev_waypt)
		weight_idx = self.features.index("laptop")
		return feature*self.weights[weight_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	# -- Distance to Human -- #

	def human_features(self, waypt, prev_waypt):
		"""
		Computes laptop cost over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions
		---
		input trajectory, output scalar feature
		"""
		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.human_dist(inter_waypt)
		return feature

	def human_dist(self, waypt):
		"""
		Computes distance from end-effector to human in xy coords
		input trajectory, output scalar distance where 
			0: EE is at more than 0.4 meters away from human
			+: EE is closer than 0.4 meters to human
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		human_xy = np.array(HUMAN_CENTER[0:2])
		dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	def human_cost(self, waypt):
		"""
		Computes the total distance to laptop cost
		---
		input trajectory, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.human_features(curr_waypt,prev_waypt)
		weight_idx = self.features.index("human")
		return feature*self.weights[weight_idx]*np.linalg.norm(curr_waypt - prev_waypt)				

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

	def coffee_constraint(self, waypt):
		"""
		Constrains orientation of robot's end-effector to be 
		holding coffee mug upright.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return EE_link.GetTransform()[:2,:3].dot([1,0,0])

		
	def coffee_constraint_derivative(self, waypt):
		"""
		Analytic derivative for coffee constraint.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
		return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(7)]).T.copy()

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
			if "table" in self.features:
				prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			if "coffee" in self.features:
				prob.AddCost(self.coffee_cost, [(t,j) for j in range(7)], "coffee%i"%t)
			if "laptop" in self.features:
				prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)
			if "human" in self.features:
				prob.AddCost(self.human_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "human%i"%t)
			if "origin" in self.features:
				prob.AddCost(self.origin_cost, [(t,j) for j in range(7)], "origin%i"%t)

		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)


	# ---- here's our algorithms for modifying the trajectory ---- #

	def learn_weights(self, u_h):
		"""
		Deforms the trajectory given human force, u_h, and
		updates features by computing difference between 
		features of new trajectory and old trajectory
		---
		input is human force and returns updated weights 
		"""
		(waypts_deform, waypts_prev) = self.deform(u_h)	
		if waypts_deform != None:
			new_features = self.featurize(waypts_deform)
			old_features = self.featurize(waypts_prev)

			Phi_p = np.array([0.0]*(len(self.features)+1))
			Phi = np.array([0.0]*(len(self.features)+1))

			# velocity feature is always there and always first
			Phi_p[0] = new_features[0]
			Phi[0] = old_features[0]

			# TODO implement summing over all features

			update_gain = np.array([0.0]*len(self.features))
			max_weight = np.array([0.0]*len(self.features))

			idx = 0
			for feat in self.features:
				update_gain[idx] = UPDATE_GAINS[feat]
				max_weight[idx] = MAX_WEIGHTS[feat]
				Phi_p[idx+1] = sum(new_features[idx+1])
				Phi[idx+1] = sum(old_features[idx+1])
				idx += 1

			#Phi_p = np.array([new_features[0], sum(new_features[1])])
			#Phi = np.array([old_features[0], sum(old_features[1])])

			update = Phi_p - Phi
			curr_weight = self.weights - update_gain*update[1:len(update)]
			curr_weight = np.clip(curr_weight, [0.0]*len(self.features), max_weight)

			print "[phri_planner] Learned weight:", curr_weight

			self.weights = curr_weight
			self.replan(self.start, self.goal, self.start_time, self.final_time, self.step_time, self.weights)
			
			return self.weights

	def deform(self, u_h):
		"""
		Deforms the next n waypoints of the upsampled trajectory
		updates the upsampled trajectory, stores old trajectory
		---
		input is human force, returns deformed and old waypts
		"""
		deform_waypt_idx = self.curr_waypt_idx + 1
		if (deform_waypt_idx + self.n) > self.num_waypts:
			return (None, None)
		waypts_prev = copy.deepcopy(self.waypts)
		waypts_deform = copy.deepcopy(self.waypts)
		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		#plotTraj(self.env, self.robot, self.bodies, self.waypts_plan, [1, 0, 0])
		return (waypts_deform, waypts_prev)
	
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

