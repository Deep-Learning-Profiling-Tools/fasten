from typing import Tuple

import torch
import triton
from dataclasses import asdict, dataclass, field

from .utils import GlobalConfig, TilingMethod


@dataclass
class BestConfig:
    tile_size: int = None  # the maximum size of each tile
    avg_tile_size: float = None  # the average size of each tile
    stddev_tile_size: float = None  # the standard deviation of tile size
    block_size: int = None  # the number of tiles belong to a block, -1: dynamic block size
    num_blocks: int = None  # number of blocks that group the tiles
    input_tiles: torch.Tensor = None
    slice_tile_mapping: torch.Tensor = None
    deterministic: bool = GlobalConfig.deterministic

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
    prune: callable = None
    record: callable = None
    cache: dict = None
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


def _compress_slices(subslices: list[list], tile_size: int, block_size: int, num_blocks: int) -> Tuple[list[list], list[list]]:
    """Compress subslices into large and small blocks."""
    compressed_subslices = []
    small_subslices = []
    for i in range(num_blocks):
        block_start_idx = i * block_size
        block_end_idx = min((i + 1) * block_size, len(subslices))

        # Extract the first and last subslice for comparison
        first_subslice = subslices[block_start_idx]
        last_subslice = subslices[block_end_idx - 1] if block_end_idx - 1 < len(subslices) else None

        # Determine if we can create a large block
        if last_subslice and first_subslice[1] == last_subslice[1] \
                and first_subslice[2] + tile_size * block_size == last_subslice[3]:
            compressed_subslices.append([first_subslice[0], first_subslice[1], first_subslice[2], last_subslice[3], 0])
        else:
            next_id = 0
            for j in range(block_start_idx, block_end_idx):
                subslice = subslices[j]
                # Set continuation index for small blocks
                if j == block_start_idx:
                    subslice[4] = len(small_subslices) + num_blocks
                    next_id = subslice[4] + 1
                    compressed_subslices.append(subslice)
                else:
                    subslice[4] = next_id
                    next_id += 1
                    small_subslices.append(subslice)

    return compressed_subslices, small_subslices


def tiling(slices: list[tuple], tile_size: int, block_size: int, reorder: bool) -> Tuple[list[list], int]:
    """Create subslices based on the tile size and compress them into blocks."""
    # Generate subslices
    subslices = [
        [index, type, off, min(off + tile_size, end), -1]
        for index, type, start, end, _ in slices
        for off in range(start, end, tile_size)
    ]

    if reorder:
        # Calculate the number of blocks
        num_blocks = triton.cdiv(len(subslices), block_size)

        # Compress subslices into large and small blocks
        compressed_subslices, small_subslices = _compress_slices(subslices, tile_size, block_size, num_blocks)

        # Combine all subslices and return
        compressed_subslices.extend(small_subslices)
        return compressed_subslices, num_blocks
    else:
        blocks = []
        cur_block = []

        def append_block():
            if cur_block[0][2] + tile_size * block_size == cur_block[-1][3]:
                blocks.append([cur_block[0][0], cur_block[0][1], cur_block[0][2], cur_block[-1][3], 0])
            else:
                blocks.append([cur_block[0][0], cur_block[0][1], cur_block[0][2], cur_block[-1][3], -1])

        for subslice in subslices:
            if len(cur_block) == block_size or (len(cur_block) > 0 and subslice[1] != cur_block[-1][1]):
                append_block()
                cur_block = []
            cur_block.append(subslice)
        if len(cur_block) > 0:
            append_block()
        return blocks, len(blocks)


def _init_segment_matmul_forward_scheduler():
    def get_key(input: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))  # (K, N)

    def prune(input_tiles, key: Tuple, config: Tuple) -> bool:
        tile_size, tiling_method, block_size = config
        if tile_size >= 64 and block_size >= 4:
            # low cache utilization
            return True
        if key[1] >= 128 and tile_size <= 32:
            # When K is large, we should use larger tile size
            return True
        return False

    return Scheduler(get_key=get_key, tile_sizes=[32, 64, 128], tiling_methods=[TilingMethod.BALANCED], block_sizes=[1, 2, 4, 8], prune=prune)


def _init_segment_matmul_backward_input_scheduler():
    def get_key(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2))  # (K, N)

    def prune(input_tiles, key: Tuple, config: Tuple) -> bool:
        tile_size, tiling_method, block_size = config
        if tile_size >= 64 and block_size >= 4:
            # low cache utilization
            return True
        if key[1] >= 128 and tile_size <= 32:
            # When K is large, we should use larger tile size
            return True
        return False

    return Scheduler(get_key=get_key, tile_sizes=[32, 64, 128], tiling_methods=[TilingMethod.BALANCED], block_sizes=[1, 2, 4, 8], prune=prune)


def _init_segment_matmul_backward_other_scheduler():
    def get_key(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor):
        return (input.size(1), other.size(2), GlobalConfig.deterministic)  # (K, N)

    def prune(input_tiles, key: Tuple, config: Tuple) -> bool:
        tile_size, tiling_method, block_size = config
        stddev_tile_size = input_tiles.stddev_tile_size
        avg_tile_size = input_tiles.avg_tile_size
        num_slices = len(input_tiles)
        if num_slices < 128 and block_size >= 2:
            # 1. low parallelism
            return True
        if block_size != 1 and stddev_tile_size / avg_tile_size >= 0.5:
            # 2. low utilization
            return True
        return False

    # Only default tiling method is supported
    return Scheduler(get_key=get_key, tile_sizes=[32, 64, 128], tiling_methods=[TilingMethod.DEFAULT], block_sizes=[1, 2, 4, 8, 16], prune=prune)


schedulers = {
    'segment_matmul_forward': _init_segment_matmul_forward_scheduler(),
    'segment_matmul_backward_input': _init_segment_matmul_backward_input_scheduler(),
    'segment_matmul_backward_other': _init_segment_matmul_backward_other_scheduler(),
}


def set_deterministic(deterministic: bool):
    GlobalConfig.deterministic = deterministic
