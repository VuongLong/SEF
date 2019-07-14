import torch
import numpy as np
from arguments import args
from common import Tensor, tensor
from env_configs import env_configs


def discount_rewards(env, rewards, states, nexts, task, current_value):
    discounted_episode_rewards = np.zeros_like(rewards)
    next_value = 0.0
    if rewards[-1] != 1:
        state_index = nexts[-1][0] + (nexts[-1][1] - 1) * env.bounds_x[1] - 1
        next_value = current_value[task, state_index]

    for i in reversed(range(len(rewards))):
        next_value = rewards[i] + args.gamma * next_value
        discounted_episode_rewards[i] = next_value

    return discounted_episode_rewards.tolist()


def statistic(env, states, actions, drewards, GAEs, next_states, current_value):
    state_dict = {}
    count_dict = {}
    for task in range(env.num_task):
        for i, (state, action, actual_value, gae, next_state) in enumerate(
                zip(states[task], actions[task], drewards[task], GAEs[task], next_states[task])):
            state_index = state[0] + (state[1] - 1) * env.bounds_x[1] - 1
            advantage = actual_value - current_value[task, state_index]
            if args.use_gae:
                advantage = gae
            #############################################################################################################
            if state_dict.get(state_index, -1) == -1:
                state_dict[state_index] = []
                count_dict[state_index] = []
                for tidx in range(env.num_task):
                    state_dict[state_index].append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    count_dict[state_index].append([0, 0, 0, 0, 0, 0, 0, 0])

            state_dict[state_index][task][action] += advantage
            count_dict[state_index][task][action] += 1
    return state_dict, count_dict


def GAE(env, rewards, states, nexts, task, current_value):
    ep_GAE = np.zeros_like(rewards)
    TD_error = np.zeros_like(rewards)
    args.gae_lambda = 0.96

    next_value = 0.0
    if rewards[-1] != 1:
        state_index = nexts[-1][0] + (nexts[-1][1] - 1) * env.bounds_x[1] - 1
        next_value = current_value[task, state_index]

    for i in reversed(range(len(rewards))):
        state_index = states[i][0] + (states[i][1] - 1) * env.bounds_x[1] - 1

        TD_error[i] = rewards[i] + args.gamma * next_value - current_value[task, state_index]
        next_value = current_value[task, state_index]

    ep_GAE[len(rewards) - 1] = TD_error[len(rewards) - 1]
    weight = args.gamma * args.gae_lambda
    for i in reversed(range(len(rewards) - 1)):
        ep_GAE[i] += TD_error[i] + weight * ep_GAE[i + 1]

    return ep_GAE.tolist()


def get_state_sharing_info(env, state, state_index, oracle_sharing, current_oracle=None):
    state_sharing_info = []
    if oracle_sharing:
        share_info = bin(env.MAP[state[1]][state[0]])[2:].zfill(env.num_task * env.num_task)
        for tidx in range(env.num_task):
            state_sharing_info.append([])
            share_info_task = share_info[tidx * env.num_task:(tidx + 1) * env.num_task]
            for otidx in range(env.num_task):
                if share_info_task[otidx] == '1':
                    state_sharing_info[tidx].append(1)
                else:
                    state_sharing_info[tidx].append(0)
    else:
        for tidx in range(env.num_task):
            state_sharing_info.append([])
            for otidx in range(env.num_task):
                state_sharing_info[tidx].append(0)
            state_sharing_info[tidx][tidx] = 1

        for tidx in range(env.num_task - 1):
            for otidx in range(tidx + 1, env.num_task):
                share_action = np.random.choice(range(len(current_oracle[otidx, tidx, state_index])),
                                                p=np.array(current_oracle[otidx, tidx, state_index]) / sum(
                                                    current_oracle[otidx, tidx, state_index]))
                state_sharing_info[tidx][otidx] = share_action
                state_sharing_info[otidx][tidx] = share_action

    return state_sharing_info


def calculate_mean_pi_sa_task(env, state, action, state_index, share_dict, count_dict, policy, Z, oracle_sharing):
    mean_pi_sa = []
    if oracle_sharing:
        for tidx in range(env.num_task):
            mean_policy_action = 0.0
            count = 0.0
            for otidx in range(env.num_task):
                if share_dict[state_index][tidx][otidx] == 1:
                    if otidx == tidx or count_dict[state_index][otidx][action] > 0:
                        mean_policy_action += (policy[otidx, state_index][action])
                        count += 1
            mean_policy_action /= count
            mean_pi_sa.append(mean_policy_action)
    else:
        for tidx in range(env.num_task):
            mean_policy_action = 0.0
            count = 0.0
            for otidx in range(env.num_task):
                if otidx == tidx or count_dict[state_index][otidx][action] > 0:
                    mean_policy_action += (Z[otidx, tidx, state_index][1] * policy[otidx, state_index][action])
                    count += Z[otidx, tidx, state_index][1]
            mean_policy_action /= count
            mean_pi_sa.append(mean_policy_action)
    return mean_pi_sa


def get_current_policy(env, policy, value, z):
    with torch.no_grad():
        env_config = env_configs[args.env]
        current_policy = Tensor(env.num_task, env_config.Policy[0], env_config.Policy[-1])
        current_value = Tensor(env.num_task, env_config.Value[0])
        current_oracle = Tensor(env.num_task, env.num_task, env_config.Z[0], env_config.Z[-1])

        for task in range(env.num_task):
            current_policy[task] = policy[task](env.cv_state_onehot)
            current_value[task] = value[task](env.cv_state_onehot).squeeze()
            current_oracle[task, task] = Tensor([0.0, 1.0])

        for i in range(env.num_task - 1):
            for j in range(i + 1, env.num_task):
                current_oracle[i, j] = z[i, j](env.cv_state_onehot)
                current_oracle[j, i] = current_oracle[i, j]

        return current_policy.cpu().numpy(), current_value.cpu().numpy(), current_oracle.cpu().numpy()
