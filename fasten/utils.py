import os
from enum import Enum

import torch
import triton.language as tl

from .operators import torch_ops, triton_ops


class GlobalConfig:
    deterministic: bool = True
    with_autotune: bool = True
    with_perf_model: bool = False
    binning_interval: float = 32.0


class TilingMethod(Enum):
    DEFAULT = 'default'
    BALANCED = 'balanced'


class Engine(Enum):
    '''
    Engine is an enum class, including 'torch', 'triton', 'auto', and the default is 'auto':
        - 'auto': use triton operators if available, otherwise use torch native operators or triton operators
        offer lower performance
        - 'torch': use torch native operators
        - 'triton': use triton operators
    '''
    TORCH = 'torch'
    TRITON = 'triton'
    AUTO = 'auto'


engine_ops = {
    Engine.TORCH: torch_ops,
    Engine.TRITON: triton_ops,
}


def torch_dtype_to_triton_dtype(dtype, grad: bool = False):
    type_dict = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
    }
    promo_type_dict = {
        torch.float16: tl.float32,
        torch.float32: tl.float32,
    }
    if grad:
        if dtype not in promo_type_dict:
            raise ValueError(f'Unsupported dtype {dtype}')
        return promo_type_dict[dtype]
    else:
        if dtype not in type_dict:
            raise ValueError(f'Unsupported dtype {dtype}')
        return type_dict[dtype]


def is_debug():
    FLAG = os.environ.get('FASTEN_DEBUG', '0')
    return FLAG == '1' or FLAG.lower() == 'true' or FLAG.lower() == 'yes' or FLAG.lower() == 'on'


def binning(x, interval: float = GlobalConfig.binning_interval):
    return x // interval
