#! /usr/bin/env python
import numpy as np
import math

class Planner(object):
	"""
	Base planner class. 
	"""

	def __init__(self):
		# important internal variables
		self.start_time = None
		self.final_time = None
		self.curr_waypt_idx = None

		# these variables are for trajopt
		self.waypts_plan = None
		self.num_waypts_plan = None
		self.step_time_plan = None

		# these variables are for the upsampled trajectory
		self.waypts = None
		self.num_waypts = None
		self.step_time = None
		self.waypts_time = None

		self.start = None
		self.goal = None

	def replan(self, start, goal, start_time, final_time, step_time):
		"""
		Replan a straight-line trajectory from start to goal.
		---
		input trajectory parameters, update raw and upsampled trajectories
		"""
		self.start = start
		self.goal = goal
		self.start_time = start_time
		self.final_time = final_time
		self.curr_waypt_idx = 0
		self.num_waypts_plan = 2 	# straight-line traj is default
		self.waypts_plan = [start, goal]
		self.step_time_plan	= (self.final_time - self.start_time)/(self.num_waypts_plan - 1)	
		self.upsample(step_time)

		# return the upsampled waypts
		return self.waypts

	def upsample(self, step_time):
		"""
		Put waypoints along trajectory at step_time increments.
		---
		input desired time increment, update upsampled trajectory
		"""
		num_waypts = int(math.ceil((self.final_time - self.start_time)/step_time)) + 1
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts
		
		t = self.start_time
		for i in range(num_waypts):
			if t >= self.final_time:
				waypts_time[i] = self.final_time
				waypts[i,:] = self.waypts_plan[self.num_waypts_plan - 1]
			else:
				deltaT = t - self.start_time
				prev_idx = int(deltaT/self.step_time_plan)
				prev = self.waypts_plan[prev_idx]
				next = self.waypts_plan[prev_idx + 1]
				waypts_time[i] = t
				waypts[i,:] = prev+((t-prev_idx*self.step_time_plan)/self.step_time_plan)*(next-prev)
			t += step_time
		self.step_time = step_time
		self.num_waypts = num_waypts
		self.waypts = waypts
		self.waypts_time = waypts_time

	def downsample(self):
		"""
		Updates the trajopt trajectory from the upsampled trajectory.
		changes the trajopt waypoints between start and goal.
		"""
		for index in range(1,self.num_waypts_plan-1):
			t = self.start_time + index*self.step_time_plan
			target_pos = self.interpolate(t)
			self.waypts_plan[index,:] = target_pos.reshape((1,7))
			
	def interpolate(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		if curr_time >= self.final_time:
			self.curr_waypt_idx = self.num_waypts - 1
			target_pos = self.waypts[self.curr_waypt_idx]
		else:
			deltaT = curr_time - self.start_time
			self.curr_waypt_idx = int(deltaT/self.step_time)
			prev = self.waypts[self.curr_waypt_idx]
			next = self.waypts[self.curr_waypt_idx + 1]
			ti = self.waypts_time[self.curr_waypt_idx]
			tf = self.waypts_time[self.curr_waypt_idx + 1]
			target_pos = (next - prev)*((curr_time-ti)/(tf - ti)) + prev		
		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos

