import argparse
from torch import nn, optim


class Arguments(argparse.Namespace):
    # Model arguments
    optimizer = optim.RMSprop
    lr = 1e-3
    hidden_activation = nn.ReLU
    confident_boundary = 0.3

    # Env arguments
    gamma = 0.99
    gae_lambda = 0.96
    use_gae = True
    env = 'GridWorld'

    # Policy
    rollouts = 8
    epochs = 1000
    oracle_sharing = False

args = Arguments()
