from typing import Tuple

import torch
from dataclasses import asdict, dataclass, field

from .utils import TilingMethod


@dataclass
class BestConfig:
    tile_size: int = None
    num_blocks: int = None
    input_tiles: torch.Tensor = None

    def asdict(self):
        return asdict(self)


@dataclass
class CacheEntry:
    best_ms: float
    best_config: BestConfig
    best_op: callable


@dataclass
class Scheduler:
    get_key: callable
    default_tile_size: int = 32
    tile_sizes: list[int] = field(default_factory=lambda: [32])
    default_tiling_method = TilingMethod.DEFAULT
    tiling_methods: list[TilingMethod] = field(default_factory=lambda: [TilingMethod.DEFAULT])


def default_tiling(slices: list, tile_size: int) -> Tuple[list, int]:
    subslices = []
    for slice in slices:
        index = slice[0]
        type = slice[1]
        start = slice[2]
        end = slice[3]
        for off in range(start, end, tile_size):
            subslices.append([index, type, off, min(off + tile_size, end), -1])
    return subslices, len(subslices)


def balance_tiling(slices: list, tile_size: int, large_tile_size: int, subslices: list) -> Tuple[list, int]:
    subslices = []
    for slice in slices:
        index = slice[0]
        type = slice[1]
        start = slice[2]
        end = slice[3]
        for off in range(start, end, large_tile_size):
            if off + large_tile_size <= end:
                subslices.append([index, type, off, off + large_tile_size, -1])
            else:
                for sub_off in range(off, end, tile_size):
                    subslices.append([index, type, sub_off, min(sub_off + tile_size, end), -1])

    return subslices, len(subslices)


def _init_segment_matmul_forward_scheduler():
    def get_key(input: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))
    return Scheduler(get_key=get_key, tile_sizes=[16, 32, 64, 128], tiling_methods=[TilingMethod.DEFAULT, TilingMethod.BALANCE])


def _init_segment_matmul_backward_scheduler():
    def get_key(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))
    return Scheduler(get_key=get_key, tile_sizes=[16, 32, 64, 128])


schedulers = {
    'segment_matmul_forward': _init_segment_matmul_forward_scheduler(),
    'segment_matmul_backward': _init_segment_matmul_backward_scheduler(),
}
