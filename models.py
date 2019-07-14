import torch
import torch.nn as nn
import torch.nn.functional as F
from common import Tensor, USE_CUDA
from arguments import args
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self, net_specs, output_layer=None):
        super().__init__()

        layers = []
        for j in range(1, len(net_specs)):
            dim_in = net_specs[j - 1]
            dim_out = net_specs[j]
            layers.append(nn.Linear(dim_in, dim_out))
            if j < len(net_specs) - 1:
                layers.append(args.hidden_activation())

        if output_layer:
            layers.append(output_layer())

        self.model = nn.Sequential(*layers)

        self._optimizer = args.optimizer(self.parameters(), lr=args.lr)
        if USE_CUDA:
            self.cuda()

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x

    def _loss(self, outputs, targets, actions=None):
        return 0.5 * F.mse_loss(outputs, targets)

    def update_parameters(self, inputs, targets, actions=None):
        outputs = self(inputs)
        loss = self._loss(outputs, targets, actions)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss


class StochasticPiNet(Net):
    def __init__(self, net_specs):
        super().__init__(net_specs, nn.Softmax)

    # targets of Policy include [actions, reward]
    def _loss(self, outputs, rewards, actions):
        m = Categorical(outputs)
        neg_log_prob = -m.log_prob(actions)


        return (neg_log_prob * rewards).mean()



