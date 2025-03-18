import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch_mlu

import xpu_graph
from xpu_graph.config import OptLevel

import random
import numpy as np
import torch.multiprocessing as mp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mlu.manual_seed(seed)
    torch.mlu.manual_seed_all(seed)

set_seed(12)


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.len


class MatMulModel(nn.Module):
    def __init__(self, in_features=10):
        super(MatMulModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, in_features))
        self.bias = nn.Parameter(torch.randn(in_features))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


def train_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("cncl", rank=rank, world_size=world_size)
    torch.mlu.set_device(rank)


def cleanup():
    dist.destroy_process_group()
