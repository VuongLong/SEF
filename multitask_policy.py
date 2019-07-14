import torch
import numpy as np           # Handle matrices
from rollout import Rollout
from arguments import args
from utils import GAE, discount_rewards, get_state_sharing_info, calculate_mean_pi_sa_task, statistic, get_current_policy
from common import Tensor, tensor
from torch.distributions import Categorical


class MultitaskPolicy(object):
	def __init__(
			self,
			env,
			env_config,
			policy,
			value,
			oracle_network,
			share_exp,
			oracle):

		self.env_config = env_config
		self.PGNetwork = policy
		self.VNetwork = value
		self.ZNetwork = oracle_network
		self.rollout = Rollout(number_episode=args.rollouts, map_index=env_config.map_index)

		self.oracle = oracle
		self.share_exp = share_exp
		self.env = env

	@staticmethod
	def _get_state_sharing_action(state_index, current_oracle, task, other_task, share_dict):
		share_action = np.random.choice(range(len(current_oracle[task, other_task, state_index])),
										p=np.array(current_oracle[task, other_task, state_index]) / sum(
											current_oracle[task, other_task, state_index]))
		if args.oracle_sharing:
			share_action = share_dict[state_index][task][other_task]
		return share_action

	@staticmethod
	def _clip_important_weight(weight):
		if weight > 1.2:
			weight = 1.2
		if weight < 0.8:
			weight = 0.8
		return weight

	def _process_PV_batch(self, states, actions, drewards, GAEs, next_states, current_policy, current_value, current_oracle, count_dict):
		batch_ss, batch_as, batch_Qs, batch_Ts = [], [], [], []
		share_ss, share_as, share_Ts = [], [], []
		for task in range(self.env.num_task):
			batch_ss.append([])
			batch_as.append([])
			batch_Qs.append([])
			batch_Ts.append([])
			share_ss.append([])
			share_as.append([])
			share_Ts.append([])

		share_dict = {}
		mean_sa_dict = {}

		for current_task in range(self.env.num_task):
			for i, (state, action, actual_value, gae, next_state) in enumerate(zip(states[current_task], actions[current_task], drewards[current_task], GAEs[current_task], next_states[current_task])):
				state_index = state[0]+(state[1]-1)*self.env.bounds_x[1]-1

				advantage = actual_value - current_value[task, state_index]
				if args.use_gae:
					advantage = gae	
					
				if self.share_exp:
					if share_dict.get(state_index, -1) == -1:
						share_dict[state_index] = get_state_sharing_info(self.env, state, state_index, args.oracle_sharing, current_oracle)
					if mean_sa_dict.get((state_index, action), -1) == -1:
						mean_sa_dict[state_index, action] = calculate_mean_pi_sa_task(self.env, state, action, state_index, share_dict, count_dict, current_policy, current_oracle, args.oracle_sharing)

					for other_task in range(self.env.num_task):

						share_action = MultitaskPolicy._get_state_sharing_action(state_index, current_oracle, current_task, other_task, share_dict)
						if share_action == 1:
							important_weight = current_policy[other_task, state_index][action]/mean_sa_dict[state_index, action][other_task]
							clip_important_weight = MultitaskPolicy._clip_important_weight(important_weight)

							if (0.8 <= important_weight <= 1.2) or (clip_important_weight*advantage > important_weight * advantage):
								if other_task == current_task:
									# use current_task's experience to update itself
									batch_ss[current_task].append(self.env.cv_state_onehot[state_index].tolist())
									batch_as[current_task].append(self.env.cv_action_onehot[action].tolist())
									batch_Qs[current_task].append(actual_value)
									batch_Ts[current_task].append(important_weight*advantage)
								else:
									# use current_task's experience to update other_task
									share_ss[other_task].append(self.env.cv_state_onehot[state_index].tolist())
									share_as[other_task].append(self.env.cv_action_onehot[action].tolist())
									share_Ts[other_task].append(important_weight * advantage)
				else:
					batch_ss[current_task].append(self.env.cv_state_onehot[state_index].tolist())
					batch_as[current_task].append(self.env.cv_action_onehot[action].tolist())
					batch_Qs[current_task].append(actual_value)
					batch_Ts[current_task].append(advantage)

				#############################################################################################################

		return batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts

	def _process_Z_batch(self, state_dict, count_dict):
		z_ss, z_as, z_rs = {}, {}, {}
		for i in range(self.env.num_task - 1):
			for j in range(i + 1, self.env.num_task):
				z_ss[i, j] = []
				z_as[i, j] = []
				z_rs[i, j] = []
		for v in state_dict.keys():
			for i in range(self.env.num_task - 1):
				for j in range(i + 1, self.env.num_task):
					for action in range(self.env.action_size):

						z_reward = 0.0
						if state_dict[v][i][action] * state_dict[v][j][action] > 0:
							z_reward = min(abs(state_dict[v][i][action]), abs(state_dict[v][j][action]))
							z_action = [0, 1]

						if state_dict[v][i][action] * state_dict[v][j][action] < 0:
							z_reward = min(abs(state_dict[v][i][action]), abs(state_dict[v][j][action]))
							z_action = [1, 0]

						if sum(count_dict[v][i]) == 0 and sum(count_dict[v][j]) > 0:
							z_reward = 0.001
							z_action = [1, 0]

						if sum(count_dict[v][j]) == 0 and sum(count_dict[v][i]) > 0:
							z_reward = 0.001
							z_action = [1, 0]

						if z_reward > 0.0:
							z_ss[i, j].append(self.env.cv_state_onehot[v].tolist())
							z_as[i, j].append(z_action)
							z_rs[i, j].append(z_reward)
		return z_ss, z_as, z_rs

	def _make_batch(self, epoch):
		current_policy, current_value, current_oracle = get_current_policy(self.env, self.PGNetwork, self.VNetwork, self.ZNetwork)

		# states = [
		#task1		[[---episode_1---],...,[---episode_n---]],
		#task2		[[---episode_1---],...,[---episode_n---]]
		#			]
		states, tasks, actions, rewards, next_states = self.rollout.rollout_batch(self.PGNetwork, current_policy, epoch)

		discounted_rewards, GAEs = [], []
		for task in range(self.env.num_task):
			discounted_rewards.append([])
			GAEs.append([])
			for ep_state, ep_next, ep_reward in zip(states[task], next_states[task], rewards[task]):	
				discounted_rewards[task] += discount_rewards(self.env, ep_reward, ep_state, ep_next, task, current_value)
				GAEs[task] += GAE(self.env, ep_reward, ep_state, ep_next, task, current_value)
			
			states[task] = np.concatenate(states[task])       
			tasks[task] = np.concatenate(tasks[task])     
			actions[task] = np.concatenate(actions[task])     
			rewards[task] = np.concatenate(rewards[task])
			next_states[task] = np.concatenate(next_states[task])

		state_dict, count_dict = statistic(self.env, states, actions, discounted_rewards, GAEs, next_states, current_value)
		task_states, task_actions, task_target_values, task_advantages, \
		sharing_states, sharing_actions, sharing_advantages = self._process_PV_batch(states,
																			  actions,
																			  discounted_rewards,
																			  GAEs,
																			  next_states,
																			  current_policy,
																			  current_value,
																			  current_oracle,
																			  count_dict)

		z_states, z_actions, z_rewards = self._process_Z_batch(state_dict, count_dict)

		return task_states, task_actions, task_target_values, task_advantages, \
			   sharing_states, sharing_actions, sharing_advantages, \
			   np.concatenate(rewards), \
			   z_states, z_actions, z_rewards

	def update_value_function(self, states, target_value):
		for t in range(self.env.num_task):
			self.VNetwork[t].update_parameters(Tensor(states[t]), Tensor(target_value[t]))

	def update_sharing_Z_agent(self, states, actions, rewards):
		for i in range(self.env.num_task - 1):
			for j in range(i + 1, self.env.num_task):
				if len(states[i, j]) > 0:
					self.ZNetwork[i, j].update_parameters(Tensor(states[i, j]), Tensor(rewards[i, j]), Tensor(actions[i, j]))

	def update_policy(self, states, actions, advantage):
		for t in range(self.env.num_task):
			self.PGNetwork[t].update_parameters(Tensor(states[t]), Tensor(advantage[t]), Tensor(actions[t]))

	def train(self):
		evarage_samples = 0
		for epoch in range(args.epochs):
			print(epoch)
			task_states, task_actions, task_target_values, task_advantages, \
			sharing_states, sharing_actions, sharing_advantages, \
			recorded_rewards, \
			z_states, z_actions, z_rewards = self._make_batch(epoch)

			self.update_value_function(task_states,  task_target_values)

			if self.share_exp:
				self.update_sharing_Z_agent(z_states, z_actions, z_rewards)

				for task_index in range(self.env.num_task):
					task_states[task_index] += sharing_states[task_index]
					task_actions[task_index] += sharing_actions[task_index]
					task_advantages[task_index] += sharing_advantages[task_index]

			self.update_policy(task_states, task_actions, task_advantages)

			# WRITE TF SUMMARIES
			evarage_samples += len(recorded_rewards) / self.env.num_task
			total_reward_of_that_batch = np.sum(recorded_rewards) / self.env.num_task
			mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, args.rollouts)
