from enum import Enum

import torch
import triton.language as tl

from .operators import torch_ops, triton_ops


class TilingMethod(Enum):
    DEFAULT = 'default'


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


def torch_dtype_to_triton_dtype(dtype):
    type_dict = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
    }
    if dtype not in type_dict:
        raise ValueError(f'Unsupported dtype {dtype}')
    return type_dict[dtype]
