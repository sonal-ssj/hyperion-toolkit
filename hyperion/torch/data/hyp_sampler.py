import logging
import math

import numpy as np
import torch
import torch.distributed as dist
from jsonargparse import ActionParser, ArgumentParser
from torch.utils.data import Sampler


class HypSampler(Sampler):
    def __init__(self, shuffle=False, seed=1234):
        super().__init__(None)
        self.epoch = 0
        self.batch = 0
        self.init_batch = 0
        self.shuffle = shuffle
        self.seed = seed

        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        self.rank = rank
        self.world_size = world_size
        self.rng = torch.Generator()

    def set_epoch(self, epoch, batch=0):
        self.epoch = epoch
        self.init_batch = batch

    def _set_seed(self):
        if self.shuffle:
            self.rng.manual_seed(self.seed + 10 * self.epoch + 100 * self.init_batch)
        else:
            self.rng.manual_seed(self.seed)

    def __iter__(self):
        self.batch = self.init_batch
        self.init_batch = 0
        self._set_seed()
        return self