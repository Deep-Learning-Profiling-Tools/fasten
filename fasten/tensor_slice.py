from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
from triton.testing import do_bench

from .operators import torch_ops, triton_ops
from .scheduler import (BestConfig, CacheEntry, Scheduler, balanced_tiling,
                        default_tiling, schedulers)
from .utils import TilingMethod, is_debug


class TensorSlice:
    '''
        Construct a TensorSlice data structure

        Args:
            data: The original data tensor, could be on either the CPU or the GPU.
                    It must have been sorted by types.
            slices: A 5-dim PyTorch Tensor, where each row represents [type_index, type, start, end, next].
                    It can also be a int or a list, then internally we transform it to a tensor.
            device: The device to put the slices on, default is 'cpu'
    '''

    def __init__(self, data: torch.Tensor, slices: Union[torch.Tensor, list, int], device: str = 'cpu', block_size: int = 1, num_blocks: Optional[int] = None) -> None:
        self._data = data

        if type(slices) is int:
            # each slice is a single type
            self._slices = torch.zeros((slices, 5), dtype=torch.long, device='cpu')
            for i in range(0, slices):
                self._slices[i][0] = i
                self._slices[i][1] = i
                self._slices[i][2] = i
                self._slices[i][3] = i + 1
                self._slices[i][4] = -1
            self._slices = self._slices.to(device)
        elif type(slices) is list:
            # 2d list, nx5
            self._slices = torch.as_tensor(slices, dtype=torch.long, device=device)
        else:
            self._slices = slices.to(device)
        # Don't backpropagate on slice tensors
        self._slices.requires_grad = False
        self._block_size = block_size
        self._num_blocks = num_blocks if num_blocks is not None else len(self._slices)
        self._cache = dict()

    def _init_mappings(self):
        if not hasattr(self, '_type_slice_dict'):
            self._type_slice_dict = OrderedDict()
            for i in range(self._slices.size(0)):
                self._type_slice_dict[self._slices[i, 1].item()] = i
        if not hasattr(self, '_slice_type_dict'):
            self._slice_type_dict = OrderedDict()
            for i in range(self._slices.size(0)):
                self._slice_type_dict[i] = self._slices[i, 1].item()

    def __len__(self) -> int:
        return self._slices.size(0)

    @property
    def data_size(self) -> int:
        return self.stop(is_tensor=False) - self.start(is_tensor=False)

    def start(self, is_tensor: bool = True):
        '''
            Get the start index of the original tensor.

            Args:
                is_tensor: If true, return a tensor. Otherwise, return a python int.
        '''
        return self._slices[0, 2] if is_tensor else self._slices[0, 2].item()

    def stop(self, is_tensor: bool = True):
        '''
            Get the stop index of the original tensor.

            Args:
                is_tensor: If true, return a tensor. Otherwise, return a python int.
        '''
        return self._slices[-1, 3] if is_tensor else self._slices[-1, 3].item()

    @property
    def slices(self):
        return self._slices

    @property
    def data(self):
        return self._data

    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def block_size(self):
        return self._block_size

    def get_slice_from_type(self, type: int, is_tensor: bool = True):
        '''
            Get the slice of the original tensor from the type.

            Args:
                type: The type
                is_tensor: If true, return a tensor. Otherwise, return a python slice.
        '''
        self._init_mappings()
        entry = self._slices[self._type_slice_dict[type]][2:4]
        return entry if is_tensor else slice(entry[0].item(), entry[1].item())

    def get_slice_from_index(self, index: int, is_tensor: bool = True):
        '''
            Get the slice of the original tensor from the slice index.

            Args:
                index: The slice index
                is_tensor: If true, return a tensor. Otherwise, return a python slice.
        '''
        self._init_mappings()
        entry = self._slices[index][2:4]
        return entry if is_tensor else slice(entry[0].item(), entry[1].item())

    def get_type_from_index(self, index: int, is_tensor: bool = True) -> int:
        '''
            Get the type from the slice index.

            Args:
                index: The slice index
                is_tensor: If true, return a tensor. Otherwise, return a python int.
        '''
        self._init_mappings()
        return self._slices[index][1] if is_tensor else self._slices[index][1].item()

    def _lookup_cache(self, op_name: str, key: tuple) -> CacheEntry:
        if op_name in self._cache and key in self._cache[op_name]:
            return self._cache[op_name][key]
        return None

    def _update_cache(self, op_name: str, key: tuple, entry: CacheEntry):
        if op_name not in self._cache:
            self._cache[op_name] = dict()
        self._cache[op_name][key] = entry

    def tiling(self, tile_size: int = Scheduler.default_tile_size, block_size: int = Scheduler.default_block_size, method: TilingMethod = Scheduler.default_tiling_method):
        assert tile_size > 0
        slices = self._slices.tolist()
        num_blocks = None
        if method == TilingMethod.DEFAULT:
            subslices, num_blocks = default_tiling(slices, tile_size, block_size)
        elif method == TilingMethod.BLOCKED:
            subslices, num_blocks = balanced_tiling(slices, tile_size, block_size)
        else:
            raise ValueError(f'Unsupported tiling method {method}')
        return TensorSlice(self.data, subslices, self._slices.device, block_size=block_size, num_blocks=num_blocks)

    def schedule(self, op_name: str, *args, autotune: bool = False) -> CacheEntry:
        scheduler = schedulers[op_name]
        key = scheduler.get_key(*args)
        cache_entry = self._lookup_cache(op_name, key)

        if cache_entry is not None:
            return cache_entry

        if autotune:
            best_ms, best_config, best_op = self.autotune(op_name, *args, scheduler=scheduler)
        else:
            best_ms, best_config, best_op = self.use_defaults(op_name, scheduler=scheduler)

        cache_entry = CacheEntry(best_ms, best_config, best_op)
        self._update_cache(op_name, key, cache_entry)
        return cache_entry

    def autotune(self, op_name: str, *args, scheduler: Scheduler) -> Tuple[float, BestConfig, callable]:
        best_op = getattr(torch_ops, op_name)
        best_ms = do_bench(lambda: best_op(*args, input_slices=self.slices), warmup=5, rep=10)
        best_config = BestConfig()
        debug = is_debug()

        triton_op = getattr(triton_ops, op_name)
        for tile_size, tiling_method, block_size in scheduler.get_configs():
            input_tiles = self.tiling(tile_size, method=tiling_method, block_size=block_size)
            ms = do_bench(
                lambda: triton_op(
                    *args,
                    input_slices=self.slices,
                    input_tiles=input_tiles.slices,
                    num_blocks=input_tiles.num_blocks,
                    block_size=block_size,
                    tile_size=tile_size
                ),
                warmup=1 if debug else 5,
                rep=1 if debug else 10
            )
            if debug:
                print(f'op_name={op_name}, tile_size={tile_size}, block_size={block_size}, tiling_method={tiling_method}, ms={ms}')
            if ms < best_ms:
                best_ms, best_op, best_config = ms, triton_op, BestConfig(tile_size=tile_size, block_size=block_size, input_tiles=input_tiles.slices, num_blocks=input_tiles.num_blocks)

        return best_ms, best_config, best_op

    def use_defaults(self, op_name: str, scheduler: Scheduler) -> Tuple[float, BestConfig, callable]:
        input_tiles = self.tiling(scheduler.default_tile_size, method=scheduler.default_tiling_method, block_size=scheduler.default_block_size)
        return 0.0, BestConfig(tile_size=scheduler.default_tile_size, block_size=scheduler.default_block_size, input_tiles=input_tiles.slices, num_blocks=input_tiles.num_blocks), getattr(triton_ops, op_name)


def compact_tensor_types(data: torch.Tensor, types: torch.Tensor, *,
                         dim: int = 0, descending: bool = False,
                         is_sorted: bool = False, device: str = 'cpu') -> TensorSlice:
    """
    Sort the types and its corresponding tensor, if given

    Args:
        data (torch.Tensor): The input data to be sorted.
        types (torch.Tensor): The type of each record.
        dim (int, optional): Which dimension of the tensor represents types. Defaults to 0.
        descending (bool, optional): If true, sort the tensor in descending order. Defaults to False.
        is_sorted (bool, optional): If true, assumes types is already sorted. Defaults to False.
        device (str, optional): The device to put the slices. Note that tensor and sorted_types remain on the original device. Defaults to 'cpu'.

    Returns:
        TensorSlice: The sorted tensor and its corresponding TensorSlice.
    """
    if not is_sorted:
        sorted_types, type_indices = torch.sort(types, descending=descending, stable=True)
    else:
        sorted_types = types

    unique_types, type_counts = torch.unique_consecutive(
        sorted_types, return_inverse=False, return_counts=True)

    type_list = []
    cur_index = 0
    for i in range(len(unique_types)):
        type_list.append([
            i, unique_types[i].item(), cur_index, cur_index + type_counts[i].item(), -1])
        cur_index += type_counts[i].item()

    sorted_data = torch.index_select(data, dim=dim, index=type_indices) if not is_sorted else data
    return TensorSlice(sorted_data, type_list, device=device)
