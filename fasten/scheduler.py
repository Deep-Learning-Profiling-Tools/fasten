from typing import Tuple

import torch
import triton
from dataclasses import asdict, dataclass, field

from .utils import TilingMethod


@dataclass
class BestConfig:
    tile_size: int = None  # the maximum size of each tile
    block_size: int = None  # the number of tiles belong to a block, -1: dynamic block size
    num_blocks: int = None  # number of blocks that group the tiles
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
    tile_sizes: list[int] = field(default_factory=lambda: [Scheduler.default_block_size])
    default_tiling_method = TilingMethod.DEFAULT
    tiling_methods: list[TilingMethod] = field(default_factory=lambda: [Scheduler.default_tiling_method])
    default_block_size: int = 1
    block_sizes: list[int] = field(default_factory=lambda: [Scheduler.default_block_size])

    def get_configs(self):
        configs = []
        for tile_size in self.tile_sizes:
            for tiling_method in self.tiling_methods:
                for block_size in self.block_sizes:
                    configs.append((tile_size, tiling_method, block_size))
        return configs


def default_tiling(slices: list, tile_size: int, block_size: int) -> Tuple[list, int]:
    subslices = []
    for slice in slices:
        index = slice[0]
        type = slice[1]
        start = slice[2]
        end = slice[3]
        for off in range(start, end, tile_size):
            subslices.append([index, type, off, min(off + tile_size, end), -1])
    num_blocks = triton.cdiv(len(subslices), block_size)
    for i in range(num_blocks):
        last_subslice_idx = (i + 1) * block_size - 1
        if last_subslice_idx >= len(subslices):
            continue
        first_subslice = subslices[i * block_size]
        last_subslice = subslices[last_subslice_idx]
        first_subslice_start = first_subslice[2]
        last_subslice_end = last_subslice[3]
        first_subslice_type = first_subslice[1]
        last_subslice_type = last_subslice[1]
        if first_subslice_type == last_subslice_type and first_subslice_start + tile_size * block_size == last_subslice_end:
            first_subslice[4] = 1  # large block
        # else small block
    return subslices, num_blocks


def _init_segment_matmul_forward_scheduler():
    def get_key(input: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))
    return Scheduler(get_key=get_key, tile_sizes=[16, 32, 64, 128], tiling_methods=[TilingMethod.DEFAULT], block_sizes=[1, 2, 4, 8, 16, 32, 64])


def _init_segment_matmul_backward_scheduler():
    def get_key(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))
    return Scheduler(get_key=get_key, tile_sizes=[16, 32, 64, 128])


schedulers = {
    'segment_matmul_forward': _init_segment_matmul_forward_scheduler(),
    'segment_matmul_backward': _init_segment_matmul_backward_scheduler(),
}
