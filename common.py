from typing import Union, List

import torch
import numpy as np
import random

NetType = Union[List[int], torch.nn.Module, type]

USE_CUDA = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
np.set_printoptions(precision=2, linewidth=200)


def seed_libraries(random_seed):
    np.random.seed(random_seed + 1)
    torch.manual_seed(random_seed + 2)
    random.seed(random_seed + 3)


def tensor(x: np.ndarray) -> Tensor:
    return torch.from_numpy(x).type(Tensor)


def set_device(gpu: int, num_gpus: int=1):
    if gpu >= 0 and USE_CUDA:
        torch.cuda.set_device(gpu)
        multi_gpu_message = f'-{gpu + num_gpus - 1}' if num_gpus > 1 else ''
        print(f'Device: GPU {gpu}{multi_gpu_message}')
    else:
        print(f'Device: CPU')
