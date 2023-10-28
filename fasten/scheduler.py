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
    return subslices, triton.cdiv(len(subslices), block_size)


def balanced_tiling(slices: list, tile_size: int, block_size: int) -> Tuple[list, int]:
    slice_pool = []
    large_tile_size = tile_size * block_size
    for slice in slices:
        index = slice[0]
        type = slice[1]
        start = slice[2]
        end = slice[3]
        for off in range(start, end, large_tile_size):
            if off + large_tile_size <= end:
                slice_pool.append([index, type, off, off + large_tile_size, -1])
            else:
                slice_pool.append([index, type, off, end, -1])

    last_small_slice_idx = -1
    last_block_size = 0
    subslices = []
    small_slice_pool = []
    small_slice_indices = []
    for slice in slice_pool:
        slice_size = slice[3] - slice[2]
        if slice_size == large_tile_size:
            # large slice => single block
            subslices.append(slice)
        else:
            # small slice => a chain of blocks
            if last_small_slice_idx == -1 or slice_size + last_block_size > large_tile_size:
                last_small_slice_idx = len(subslices)
                last_block_size = slice_size
                subslices.append(slice)
            else:
                last_block_size += slice_size
                small_slice_pool.append(slice)
                small_slice_indices.append(last_small_slice_idx)

    num_blocks = len(subslices)
    for i, index in enumerate(small_slice_indices):
        if i or index != small_slice_indices[i - 1]:
            subslices[index][4] = len(subslices)
        else:
            subslices[i - 1][4] = len(subslices)
        subslices.append(small_slice_pool.pop(0))

    return subslices, num_blocks


def _init_segment_matmul_forward_scheduler():
    def get_key(input: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))
    return Scheduler(get_key=get_key, tile_sizes=[16, 32, 64, 128], tiling_methods=[TilingMethod.DEFAULT, TilingMethod.BALANCED], block_sizes=[1, 2, 4, 8, 16])


def _init_segment_matmul_backward_scheduler():
    def get_key(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))
    return Scheduler(get_key=get_key, tile_sizes=[16, 32, 64, 128])


schedulers = {
    'segment_matmul_forward': _init_segment_matmul_forward_scheduler(),
    'segment_matmul_backward': _init_segment_matmul_backward_scheduler(),
}
