"""
This code implements a trajectory-optimization based planner that updates
the weights on the cost function based on physical human-robot interaction (PHRI).

The weight update is adaptive based on human interaction relevance. Soon to
come publication.

Author: Andreea Bobu (abobu@berkeley.edu)
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
from scipy.optimize import minimize, newton
from scipy.stats import chi2

import trajoptpy
import or_trajopt
import openravepy
from openravepy import *

import planner
import openrave_utils
from openrave_utils import *

import logging
import copy
import os
import itertools
import pickle

# feature constacts (update gains and max weights)
UPDATE_GAINS = {'table':2.0, 'coffee':2.0, 'laptop':100.0, 'human':20.0}
MAX_WEIGHTS = {'table':1.0, 'coffee':1.0, 'laptop':10.0, 'human':10.0}

OBS_CENTER = [-1.3858/2.0 - 0.1, -0.1, 0.0]
HUMAN_CENTER = [0.0, -0.4, 0.0]

# fit a chi-squared distribution to p(beta|r); numers are [deg_of_freedom, loc, scale]
P_beta = {"table0": [1.83701582842, 0.0, 0.150583961407], "table1": [2.8, 0.0, 0.4212940611], "coffee0": [1.67451171875, 0.0, 0.05], "coffee1": [2.8169921875, 0.0, 0.3], "human0": [2.14693459432, 0.0, 0.227738059531], "human1": [5.0458984375, 0.0, 0.25]}

class BETAPlanner(planner.Planner):
	"""
	This class plans a trajectory from start to goal
	with TrajOpt.
	"""

	def __init__(self, feat_list, traj_cache=None):

		# ---- important internal variables ---- #
		super((BETAPlanner, self).__init__()
		self.feat_list = feat_list		# 'table', 'human', 'coffee', 'origin', 'laptop'
		self.num_features = len(self.feat_list)

		self.MAX_ITER = 40

		# this is the cache of trajectories computed for all max/min weights
		if traj_cache is not None:
			here = os.path.dirname(os.path.realpath(__file__))
			self.traj_cache = pickle.load( open( here + traj_cache, "rb" ) )
		else:
			self.traj_cache = None

        # variables for storing outside of the planner
		self.weights = [0.0]*self.num_features
		self.betas = [1.0]*self.num_features
		self.betas_u = [1.0]*self.num_features
		self.waypts_prev = None
		self.waypts_deform = None
		self.updates = [0.0]*self.num_features

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
		features = [self.velocity_features(waypts)]
		features += [[0.0 for _ in range(len(waypts)-1)] for _ in range(self.num_features)]
		for index in range(0,len(waypts)-1):
			for feat in range(1,self.num_features+1):
				if self.feat_list[feat-1] == 'table':
					features[feat][index] = self.table_features(waypts[index+1])
				elif self.feat_list[feat-1] == 'coffee':
					features[feat][index] = self.coffee_features(waypts[index+1])
				elif self.feat_list[feat-1] == 'human':
					features[feat][index] = self.human_features(waypts[index+1],waypts[index])
				elif self.feat_list[feat-1] == 'laptop':
					features[feat][index] = self.laptop_features(waypts[index+1],waypts[index])
				elif self.feat_list[feat-1] == 'origin':
					features[feat][index] = self.origin_features(waypts[index+1])
		return features

	def featurize_single(self, waypts, feat_idx):
		"""
		Computes the user-defined features for a given trajectory and a given feature.
		---
		input trajectory, output list of feature values
		"""
		features = [0.0 for _ in range(len(waypts)-1)]
		for index in range(0,len(waypts)-1):
			if self.feat_list[feat_idx] == 'table':
				features[index] = self.table_features(waypts[index+1])
			elif self.feat_list[feat_idx] == 'coffee':
				features[index] = self.coffee_features(waypts[index+1])
			elif self.feat_list[feat_idx] == 'human':
				features[index] = self.human_features(waypts[index+1],waypts[index])
			elif self.feat_list[feat_idx] == 'laptop':
				features[index] = self.laptop_features(waypts[index+1],waypts[index])
			elif self.feat_list[feat_idx] == 'origin':
				features[index] = self.origin_features(waypts[index+1])
		return features

	# -- Velocity -- #

	def velocity_features(self, waypts):
		"""
		Computes total velocity cost over waypoints, confirmed to match trajopt.
		---
		input waypoint, output scalar feature
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
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EEcoord_y = coords[6][1]
		EEcoord_y = np.linalg.norm(coords[6])
		return EEcoord_y

	def origin_cost(self, waypt):
		"""
		Computes the total distance from EE to base of robot cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.origin_features(waypt)
		feature_idx = self.feat_list.index('origin')
		return feature*self.weights[feature_idx]

	# -- Distance to Table -- #

	def table_features(self, waypt):
		"""
		Computes the total cost over waypoints based on 
		z-axis distance to table
		---
		input trajectory, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
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
		feature_idx = self.feat_list.index('table')
		return feature*self.weights[feature_idx]

	# -- Coffee (or z-orientation of end-effector) -- #

	def coffee_features(self, waypt):
		"""
		Computes the distance to table cost for waypoint
		by checking if the EE is oriented vertically according to pitch.
		Note: adding 1.5 to pitch to make it centered around 0
		---
		input trajectory, output scalar cost
		"""
		# get rotation transform, convert it to euler coordinates, and make sure the end effector is upright
		def mat2euler(mat):
			gamma = np.arctan2(mat[2,1], mat[2,2])
			beta = np.arctan2(-mat[2,0], np.sqrt(mat[2,1]**2 + mat[2,2]**2))
			alpha = np.arctan2(mat[1,0], mat[0,0])
			return np.array([gamma,beta,alpha])

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		R = EE_link.GetTransform()[:3,:3]
		[yaw, pitch, roll] = mat2euler(R)
		#return sum(abs(EE_link.GetTransform()[:2,:3].dot([1,0,0])))
		return (pitch + 1.5)

	def coffee_cost(self, waypt):
		"""
		Computes the total coffee (EE orientation) cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.coffee_features(waypt)
		feature_idx = self.feat_list.index('coffee')
		return feature*self.weights[feature_idx]

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
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
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
		feature_idx = self.feat_list.index('laptop')
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

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
			0: EE is at more than .7 meters away from human
			+: EE is closer than .7 meters to human
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		human_xy = np.array(HUMAN_CENTER[0:2])
		dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.7
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
		feature_idx = self.feat_list.index('human')
		return feature*self.weights[feature_idx]*np.linalg.norm(curr_waypt - prev_waypt)

	# ---- custom constraints --- #

	def table_constraint(self, waypt):
		"""
		Constrains z-axis of robot's end-effector to always be 
		above the table.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
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
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return EE_link.GetTransform()[:2,:3].dot([1,0,0])


	def coffee_constraint_derivative(self, waypt):
		"""
		Analytic derivative for coffee constraint.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]))
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
		return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(7)]).T.copy()


	# ---- here's trajOpt --- #

	def trajOptPose(self, start, goal, goal_pose):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Goal is a pose, not a configuration!
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input:
			start and goal pos, and a trajectory to seed trajopt with
		return:
			the waypts_plan trajectory
		"""

		print "I'm in trajopt_PLANNER trajopt pose!"

		# plot goal point
		#plotSphere(self.env, self.bodies, goal_pose, size=40)

		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]))
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4

		xyz_target = goal_pose
		quat_target = [1,0,0,0] # wxyz

		init_joint_target =  goal

		init_waypts = np.zeros((self.num_waypts_plan,7))
		for count in range(self.num_waypts_plan):
			init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)

		if self.traj_cache is not None:
			# choose seeding trajectory from cache if the weights match
			weights_span = [None]*self.num_features
			min_dist_w = [None]*self.num_features
			for feat in range(0,self.num_features):
				limit = MAX_WEIGHTS[self.feat_list[feat]]
				weights_span[feat] = list(np.arange(-limit, limit+.1, limit/2))
				min_dist_w[feat] = -limit

			weight_pairs = list(itertools.product(*weights_span))
			weight_pairs = [np.array(i) for i in weight_pairs]

			# current weights
			cur_w = np.array(self.weights)
			min_dist_idx = 0
			for (w_i, w) in enumerate(weight_pairs):
				dist = np.linalg.norm(cur_w - w)
				if dist < np.linalg.norm(cur_w - min_dist_w):
					min_dist_w = w
					min_dist_idx = w_i

			init_waypts = np.array(self.traj_cache[min_dist_idx])
		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"start_fixed" : True,
				"max_iter" : self.MAX_ITER
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [1.0]}
			}
			],
			"constraints": [
			{
				"type": "pose",
				"params" : {"xyz" : xyz_target,
                            "wxyz" : quat_target,
                            "link": "j2s7s300_link_7",
							"rot_coeffs" : [0,0,0],
							"pos_coeffs" : [35,35,35],
                            }
			}
			],
			#"init_info": {
            #    "type": "straight_line",
            #    "endpoint": init_joint_target.tolist()
			#}
			"init_info": {
                "type": "given_traj",
                "data": init_waypts.tolist()
			}
		}

		s = json.dumps(request)
		prob = trajoptpy.ConstructProblem(s, self.env)

		for t in range(1,self.num_waypts_plan):
			if 'coffee' in self.feat_list:
				prob.AddCost(self.coffee_cost, [(t,j) for j in range(7)], "coffee%i"%t)
			if 'table' in self.feat_list:
				prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			if 'laptop' in self.feat_list:
			    prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)
			if 'origin' in self.feat_list:
			    prob.AddCost(self.origin_cost, [(t,j) for j in range(7)], "origin%i"%t)
			if 'human' in self.feat_list:
			    prob.AddCost(self.human_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "human%i"%t)

		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "table%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)	

		print "I'm done with trajopt pose!"

		return self.waypts_plan


	def trajOpt(self, start, goal, traj_seed=None):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input is start and goal pos, updates the waypts_plan
		"""

		print "I'm in normal trajOpt!"
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]))
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4

		# --- linear interpolation seed --- #
		if traj_seed is None:
			print "using straight line!"
			init_waypts = np.zeros((self.num_waypts_plan,7))
			for count in range(self.num_waypts_plan):
				init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)
		else:
			print "using traj seed!"
			init_waypts = traj_seed

		if self.traj_cache is not None:
			# choose seeding trajectory from cache if the weights match
			weights_span = [None]*self.num_features
			min_dist_w = [None]*self.num_features
			for feat in range(0,self.num_features):
				limit = MAX_WEIGHTS[self.feat_list[feat]]
				weights_span[feat] = list(np.arange(-limit, limit+.1, limit/2))
				min_dist_w[feat] = -limit

			weight_pairs = list(itertools.product(*weights_span))
			weight_pairs = [np.array(i) for i in weight_pairs]

			# current weights
			cur_w = np.array(self.weights)
			min_dist_idx = 0
			for (w_i, w) in enumerate(weight_pairs):
				dist = np.linalg.norm(cur_w - w)
				if dist < np.linalg.norm(cur_w - min_dist_w):
					min_dist_w = w
					min_dist_idx = w_i

			init_waypts = np.array(self.traj_cache[min_dist_idx])

		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"max_iter" : self.MAX_ITER
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
			if 'coffee' in self.feat_list:
				prob.AddCost(self.coffee_cost, [(t,j) for j in range(7)], "coffee%i"%t)
			if 'table' in self.feat_list:
				prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			if 'laptop' in self.feat_list:
				prob.AddErrorCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "HINGE", "laptop%i"%t)
				prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)
			if 'origin' in self.feat_list:
				prob.AddCost(self.origin_cost, [(t,j) for j in range(7)], "origin%i"%t)
			if 'human' in self.feat_list:
				prob.AddCost(self.human_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "human%i"%t)


		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

		return self.waypts_plan


	# ---- here's our algorithms for modifying the trajectory ---- #

	def learnWeights(self, u_h):
		"""
		Deforms the trajectory given human force, u_h, and
		updates features by computing difference between 
		features of new trajectory and old trajectory
		---
		input is human force and returns updated weights 
		"""
		(waypts_deform, waypts_prev) = self.deform(u_h)	

		if waypts_deform is not None:
			self.waypts_deform = waypts_deform
			new_features = self.featurize(waypts_deform)
			old_features = self.featurize(waypts_prev)

			Phi_p = np.array([new_features[0]] + [sum(x) for x in new_features[1:]])
			Phi = np.array([old_features[0]] + [sum(x) for x in old_features[1:]])

			self.prev_features = Phi_p
			self.curr_features = Phi

			# Determine alpha and max theta
			update_gains = [0.0] * self.num_features
			max_weights = [0.0] * self.num_features
			feat_range = [0.0] * self.num_features
			for feat in range(0, self.num_features):
				update_gains[feat] = UPDATE_GAINS[self.feat_list[feat]]
				max_weights[feat] = MAX_WEIGHTS[self.feat_list[feat]]
			update = Phi_p - Phi
			self.updates = update[1:].tolist()

			# beta-adaptive method
			update = update[1:]
			Phi_p = Phi_p[1:]
			Phi = Phi[1:]

			### First obtain the original beta rationality from the optimization problem ###
			# Set up the unconstrained optimization problem:
			def u_unconstrained(u):
				# Optimized manually; lambda_u can be changed according to user preferences
				if self.feat_list[i] == 'table':
					lambda_u = 20000
				elif self.feat_list[i] == 'human':
					lambda_u = 1500
                elif self.feat_list[i] == 'coffee':	
					lambda_u = 20000
				u_p = np.reshape(u, (7,1))
				(waypts_deform_p, waypts_prev) = self.deform(u_p)
				H_features = self.featurize_single(waypts_deform_p,i)
				Phi_H = sum(H_features)
				cost = (Phi_H - Phi_p[i])**2
				return cost

			# Constrained variant of the optimization problem
			def u_constrained(u):
				cost = np.linalg.norm(u)**2
				return cost

			# Set up the constraints:
			def u_constraint(u):
				u_p = np.reshape(u, (7,1))
				(waypts_deform_p, waypts_prev) = self.deform(u_p)
				H_features = self.featurize_single(waypts_deform_p,i)
				Phi_H = sum(H_features)
				cost = (Phi_H - Phi_p[i])**2
				return cost

			# Compute what the optimal action would have been wrt every feature
			for i in range(self.num_features):
				# Compute optimal action
				# Every feature requires a different optimizer because every feature is different in scale
                # Every feature also requires a different Newton-Rapson lambda
				if self.feat_list[i] == 'table':
					u_h_opt = minimize(u_constrained, np.zeros((7,1)), method='SLSQP', constraints=({'type': 'eq', 'fun': u_constraint}), options={'maxiter': 10, 'ftol': 1e-6, 'disp': True})
					l = math.pi
				elif self.feat_list[i] == 'human':
					u_h_opt = minimize(u_unconstrained, np.zeros((7,1)), options={'maxiter': 10, 'disp': True})
                    l = 15.0
				elif self.feat_list[i] == 'coffee':
					u_h_opt = minimize(u_constrained, np.zeros((7,1)), method='SLSQP', constraints=({'type': 'eq', 'fun': u_constraint}), options={'maxiter': 10, 'ftol': 1e-6, 'disp': True})
					l = math.pi
				u_h_star = np.reshape(u_h_opt.x, (7, 1)) 

				# Compute beta based on deviation from optimal action
				beta_norm = 1.0/np.linalg.norm(u_h_star)**2
				self.betas[i] = self.num_features/(2*beta_norm*abs(np.linalg.norm(u_h)**2 - np.linalg.norm(u_h_star)**2))

				### Compute update using P(r|beta) for the beta estimate we just computed ###
				# Compute P(r|beta)
				mus1 = P_beta[self.feat_list[i]+"1"]
				mus0 = P_beta[self.feat_list[i]+"0"]
				p_r0 = chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) / (chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) + chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]))
				p_r1 = chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]) / (chi2.pdf(self.betas[i],mus0[0],mus0[1],mus0[2]) + chi2.pdf(self.betas[i],mus1[0],mus1[1],mus1[2]))

				# Newton-Rapson setup; define function, derivative, and
				# call optimization method
				def f_theta(weights_p):
				    num = p_r1*np.exp(weights_p*update[i])
				    denom = p_r0*(l/math.pi)**(self.num_features/2.0)*np.exp(-l*update[i]**2) + num
					return weights_p + update_gains[i]*num*update[i]/denom - self.weights[i]
				def df_theta(weights_p):
				    num = p_r0*(l/math.pi)**(self.num_features/2.0)*np.exp(-l*update[i]**2)
				    denom = p_r1*np.exp(weights_p*update[i])
				    return 1 + update_gains[i]*num/denom

				weight_p = newton(f_theta,self.weights[i],df_theta,tol=1e-04,maxiter=1000)
				
				num = p_r1*np.exp(weight_p*update[i])
				denom = p_r0*(l/math.pi)**(self.num_features/2.0)*np.exp(-l*update[i]**2) + num
				self.betas_u[i] = num/denom
			# Compute new weights
			curr_weight = self.weights - np.array(self.betas_u)*update_gains*update

		# clip values at max and min allowed weights
		for i in range(self.num_features):
			curr_weight[i] = np.clip(curr_weight[i], -max_weights[i], max_weights[i])

		self.weights = curr_weight.tolist()

		return self.weights

	def deform(self, u_h):
		"""
		Deforms the next n waypoints of the upsampled trajectory
		updates the upsampled trajectory, stores old trajectory
		---
		input is human force, returns deformed and old waypts
		"""
		waypts_prev = copy.deepcopy(self.waypts)
		waypts_deform = copy.deepcopy(self.waypts)
		gamma = np.zeros((self.n,7))
		deform_waypt_idx = self.curr_waypt_idx + 1

		if (deform_waypt_idx + self.n) > self.num_waypts:
			print "Deforming too close to end. Returning same trajectory"
			return (waypts_prev, waypts_prev)

		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		return (waypts_deform, waypts_prev)

	# ---- replanning, upsampling, and interpolating ---- #

	def replan(self, start, goal, weights, start_time, final_time, step_time, seed=None):
		"""
		Replan the trajectory from start to goal given weights.
		---
		input trajectory parameters, update raw and upsampled trajectories
		"""
		if weights is None:
			return
		self.start_time = start_time
		self.final_time = final_time
		self.curr_waypt_idx = 0
		self.weights = weights
		print "weights in replan: " + str(weights)

		if 'coffee' in self.feat_list:
			place_pose = [-0.46513, 0.29041, 0.69497]
			self.trajOptPose(start, goal, place_pose)
		else:
			self.trajOpt(start, goal, traj_seed=seed)

		self.upsample(step_time)

		return self.waypts_plan

	def kill_planner(self):
		"""
		Destroys openrave thread and environment for clean shutdown
		"""
		self.env.Destroy()
		RaveDestroy() # destroy the runtime
