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
    subslices = []
    large_tile_size = tile_size * block_size
    for slice in slices:
        index = slice[0]
        type = slice[1]
        start = slice[2]
        end = slice[3]
        for off in range(start, end, large_tile_size):
            if off + large_tile_size <= end:
                subslices.append([index, type, off, off + large_tile_size, -1])
            else:
                slice_pool.append([index, type, off, end])

    slice_pool = sorted(slice_pool, key=lambda s: s[3] - s[2], reverse=True)
    # bin packing is np hard, so we use a greedy algorithm here
    bins = []
    for slice in slice_pool:
        slice_length = slice[3] - slice[2]
        best_fit_bin_idx = -1
        least_space_left = float('inf')

        for i, bin in enumerate(bins):
            if bin[1] >= slice_length and bin[1] - slice_length < least_space_left:
                least_space_left = bin[1] - slice_length
                best_fit_bin_idx = i

        if best_fit_bin_idx != -1:
            bins[best_fit_bin_idx][0].append(slice)
            bins[best_fit_bin_idx][1] -= slice_length
        else:
            bins.append([[slice], large_tile_size - slice_length])

    num_blocks = len(bins) + len(subslices)
    block_idx = num_blocks

    # merge bins into subslices
    for bin in bins:
        bin_slices = bin[0]
        new_subslices = []

        for slice in bin_slices:
            index, type, start, end = slice[:4]
            for off in range(start, end, tile_size):
                if off + tile_size <= end:
                    new_subslices.append([index, type, off, off + tile_size, -1])
                else:
                    new_subslices.append([index, type, off, end, -1])

        # Link the subslices together
        for i in range(len(new_subslices) - 1):
            new_subslices[i][4] = block_idx + i

        block_idx += len(new_subslices) - 1
        subslices.extend(new_subslices)

    return subslices, num_blocks


def _init_segment_matmul_forward_scheduler():
    def get_key(input: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))
    return Scheduler(get_key=get_key, tile_sizes=[16, 32, 64, 128], tiling_methods=[TilingMethod.DEFAULT], block_sizes=[1, 2, 4, 8, 16])


def _init_segment_matmul_backward_scheduler():
    def get_key(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))
    return Scheduler(get_key=get_key, tile_sizes=[16, 32, 64, 128])


schedulers = {
    'segment_matmul_forward': _init_segment_matmul_forward_scheduler(),
    'segment_matmul_backward': _init_segment_matmul_backward_scheduler(),
}
