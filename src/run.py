#! /usr/bin/env python
import math
import sys, select, os
import thread
import argparse
import actionlib
import time
import planner
import precomputed_planner
import trajopt_planner
import phri_planner
import jaco_controller

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

import openrave_utils
from openrave_utils import *

pick = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]

if __name__ == '__main__':

	# set the start and goal pose of the robot in radians
	start = np.array(pick)*(math.pi/180.0)
	#start = np.array(pick_shelf)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	# set the start time of the traj, final time and step time (in seconds)
	start_time = 0.0
	final_time = 15.0
	step_time = 0.5

	# ---- default straight-line planner ---- #
	#planner = planner.Planner()
	#planner.replan(start, goal, start_time, final_time, step_time)	
	# --------------------------------------- #	

	# ---- pre-computed trajectory planner ---- #
	# traj_filename = "../resources/traj_example.p"
	# planner = precomputed_planner.PrecomputedPlanner()
	# planner.replan(None, None, start_time, final_time, step_time, traj_filename)	
	# ----------------------------------------- #	

	# ---- trajectory optimization planner ---- #
	# features = ["table", "coffee"]
	# planner = trajopt_planner.TrajoptPlanner(features)

	# specify the features for the cost function and their weighting
	# weights = [1, 0]

	# plan a trajectory from start to goal, with the feature weight
	# planner.replan(start, goal, start_time, final_time, step_time, weights)		
	# ----------------------------------------- #
	
	# --- learning from pHRI planner ---- # 
	features = ["table", "coffee"]
	planner = phri_planner.PHRIPlanner(features)
	weights = [0.0, 0.0]
	planner.replan(start, goal, start_time, final_time, step_time, weights)		

	# create the jaco controller
	sim_flag = False
	admittance_flag = True
	jaco_controller.JacoController(planner, sim_flag, admittance_flag)

	# plot the trajectory using functions from openrave_utils
	# plotCupTraj(planner.env, planner.robot, planner.bodies, traj, color=[0,1,0], increment=5)

	# sleep the sim for 20 seconds so the program doens't shut down
	time.sleep(20)

