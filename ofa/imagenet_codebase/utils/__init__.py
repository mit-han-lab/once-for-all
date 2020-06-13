# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import math
import numpy as np

try:
    import horovod.torch as hvd
except ImportError:
    pass
    # print('No horovod in environment')
    # import numpy as hvd

import torch
# from ofa.imagenet_codebase.utils.pytorch_utils import *

from .pytorch_modules import *
from .pytorch_utils import *
from .my_modules import *
from .flops_counter import *


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list


def list_sum(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] + list_sum(x[1:])


def list_weighted_sum(x, weights):
    if len(x) == 1:
        return x[0] * weights[0]
    else:
        return x[0] * weights[0] + list_weighted_sum(x[1:], weights[1:])


def list_mean(x):
    return list_sum(x) / len(x)


def list_mul(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] * list_mul(x[1:])


def list_join(val_list, sep='\t'):
    return sep.join([
        str(val) for val in val_list
    ])


def list_continuous_index(val_list, index):
    assert index <= len(val_list) - 1
    left_id = int(index)
    right_id = int(math.ceil(index))
    if left_id == right_id:
        return val_list[left_id]
    else:
        return val_list[left_id] * (right_id - index) + val_list[right_id] * (index - left_id)


def subset_mean(val_list, sub_indexes):
    sub_indexes = int2list(sub_indexes, 1)
    return list_mean([val_list[idx] for idx in sub_indexes])


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


# Horovod: average metrics from distributed training.
class DistributedMetric(object):
    
    def __init__(self, name):
        self.name = name
        self.sum = torch.zeros(1)[0]
        self.count = torch.zeros(1)[0]

    def update(self, val, delta_n=1):
        val = val * delta_n
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.count += delta_n

    @property
    def avg(self):
        return self.sum / self.count
    

class DistributedTensor(object):
    
    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = torch.zeros(1)[0]
        self.synced = False
    
    def update(self, val, delta_n=1):
        val = val * delta_n
        if self.sum is None:
            self.sum = val.detach()
        else:
            self.sum += val.detach()
        self.count += delta_n

    @property
    def avg(self):
        if not self.synced:
            self.sum = hvd.allreduce(self.sum, name=self.name)
            self.synced = True
        return self.sum / self.count
