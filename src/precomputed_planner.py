#! /usr/bin/env python
"""
This code allows you to run pre-computed sequences of waypoints on the robot.

Author: Andrea Bajcsy (abajcsy@berkeley.edu)
"""
import numpy as np
import time
import math
import planner
import pickle

class PrecomputedPlanner(planner.Planner):
	"""
	This class moves the robot along a pre-computed trajectory. 
	"""

	def __init__(self):

		# ---- important internal variables ---- #
		super(PrecomputedPlanner, self).__init__()
	
	def replan(self, start, goal, start_time, final_time, step_time, traj_filename):
			"""
			Reads trajectory from traj_filename for execution
			---
			input trajectory parameters, update raw and upsampled trajectories
			traj_filename 	- name of pickle file containing a saved trajectory
											  structure of pickle file is list of np.arrays where 
												rows = waypts and cols = joints
														(row1)--> [np.array([waypt1_dof1, waypt1_dof2, ...., waypt1_dof7]),
														(row2)-->  np.array([waypt2_dof1, waypt2_dof2, ...., waypt2_dof7]),
														 ....
														(rowN)-->  np.array([wayptN_dof1, wayptN_dof2, ...., wayptN_dof7])]
			"""
			self.start_time = start_time
			self.final_time = final_time
			self.curr_waypt_idx = 0

			self.waypts_plan = pickle.load(open(traj_filename, "rb" ))
			self.start = self.waypts_plan[0]
			self.goal = self.waypts_plan[-1]

			self.num_waypts_plan = len(self.waypts_plan)
			self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

			self.upsample(step_time)

			# return the upsampled waypts
			return self.waypts
