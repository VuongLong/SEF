# -*- coding: utf-8 -*-

from models import Net, StochasticPiNet

from constant import ACTION_SIZE
from multitask_policy import MultitaskPolicy
from env.terrain import Terrain
import argparse
from env_configs import env_configs
from arguments import args

NON = 1
SHARE = 2


def training(share_exp, oracle):
	env_config = env_configs[args.env]
	env = Terrain(env_config.map_index)
	
	policy = []
	value = []
	oracle_network = {}
	for i in range(env.num_task):
		policy_i = StochasticPiNet(env_config.Policy)
		policy.append(policy_i)
		value_i = Net(env_config.Value)
		value.append(value_i)

	for i in range(env.num_task-1):
		for j in range(i+1, env.num_task):
			oracle_network[i, j] = StochasticPiNet(env_config.Z)

	multitask_agent = MultitaskPolicy(env=env,
									env_config=env_config,
									policy=policy,
									value=value,
									oracle_network=oracle_network,
									share_exp=share_exp,
									oracle=oracle)
	
	multitask_agent.train()


training(share_exp=True, oracle=True)

