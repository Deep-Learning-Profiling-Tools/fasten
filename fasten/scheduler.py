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
    start_and_type: torch.Tensor = None
    end_and_next: torch.Tensor = None
    contiguous_flags: torch.Tensor = None

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
    default_trunc: bool = False
    trunc_flags: list[bool] = field(default_factory=lambda: [Scheduler.default_trunc])

    def get_configs(self):
        configs = []
        for tile_size in self.tile_sizes:
            for tiling_method in self.tiling_methods:
                for block_size in self.block_sizes:
                    for trunc in self.trunc_flags:
                        configs.append((tile_size, tiling_method, block_size, trunc))
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
            for j in range(block_start_idx, block_end_idx):
                subslice = subslices[j]
                # Set continuation index for small blocks
                subslice[4] = len(small_subslices) + num_blocks
                if j == block_start_idx:
                    compressed_subslices.append(subslice)
                else:
                    small_subslices.append(subslice)

    return compressed_subslices, small_subslices


def default_tiling(slices: list[tuple], tile_size: int, block_size: int) -> Tuple[list[list], int]:
    """Create subslices based on the tile size and compress them into blocks."""
    # Generate subslices
    subslices = [
        [index, type, off, min(off + tile_size, end), -1]
        for index, type, start, end, _ in slices
        for off in range(start, end, tile_size)
    ]

    # Calculate the number of blocks
    num_blocks = triton.cdiv(len(subslices), block_size)

    # Compress subslices into large and small blocks
    compressed_subslices, small_subslices = _compress_slices(subslices, tile_size, block_size, num_blocks)

    # Combine all subslices and return
    compressed_subslices.extend(small_subslices)
    return compressed_subslices, num_blocks


def trunc_slices(slices: torch.Tensor):
    start_and_type = torch.zeros((slices.size(0),), dtype=torch.long, device=slices.device)
    start_and_type.fill_(slices[:, 1] << 32)
    start_and_type |= slices[:, 3]
    end_and_next = torch.zeros((slices.size(0),), dtype=torch.long, device=slices.device)
    end_and_next.fill_(slices[:, 2] << 32)
    end_and_next |= slices[:, 4]
    # TODO: optimize this
    slices_cpu = slices.cpu()
    contiguous_flags = torch.zeros((triton.cdiv(slices.size(0), 32),), dtype=torch.int, device='cpu')
    for i in range(slices.size(0)):
        contiguous_flags[i // 32] |= (slices_cpu[i, 4] == 0).int() << (i % 32)
    contiguous_flags = contiguous_flags.to(slices.device)
    return start_and_type, end_and_next, contiguous_flags


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
