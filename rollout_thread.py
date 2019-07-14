from env.terrain import Terrain
import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
import math
import sys
from arguments import args


class RolloutThread(object):
	def __init__(
		self,
		network,
		task,
		start_x,
		start_y,
		policy,
		map_index):
	
		self.network = network
		self.task = task
		self.start_x = start_x
		self.start_y = start_y
		self.policy = policy
		self.env = Terrain(map_index)
		self.onehot_actions = np.identity(self.env.action_size, dtype=int)

	def rollout(self):
		states, tasks, actions, rewards, next_states = [], [], [], [], []

		self.env.resetgame(self.task, self.start_x, self.start_y)
		state = self.env.player.get_position()
		state_index = state[0] + (state[1] - 1) * self.env.bounds_x[1] - 1

		action_size = range(len(self.policy[self.task, state_index]))
		step = 1
		while True:
			step += 1
			if step > args.rollouts:
				break

			state_index = state[0] + (state[1] - 1) * self.env.bounds_x[1] - 1
			action = np.random.choice(action_size, p=np.array(self.policy[self.task, state_index]) / sum(self.policy[self.task, state_index]))  # select action w.r.t the actions prob

			reward, done = self.env.player.action(action)
			
			next_state = self.env.player.get_position()
			
			# Store results
			states.append(state)
			tasks.append(self.task)

			actions.append(action)
			rewards.append(reward)
			next_states.append(next_state)
			state = next_state
			
			if done:     
				break

		return states, tasks, actions, rewards, next_states
