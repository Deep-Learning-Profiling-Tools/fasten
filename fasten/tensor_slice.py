from collections import OrderedDict
from itertools import product
from typing import Tuple, Union

import torch
from triton.testing import do_bench

from .operators import torch_ops, triton_ops
from .scheduler import BestConfig, CacheEntry, Scheduler, schedulers
from .utils import TilingMethod


class TensorSlice:
    '''
        Construct a TensorSlice data structure

        Args:
            data: The original data tensor, could be on either the CPU or the GPU.
                    It must have been sorted by types.
            slices: A 3-dim PyTorch Tensor, where each row represents [type_index, type, start, end].
                    It can also be a int or a list, then internally we transform it to a tensor.
            device: The device to put the slices on, default is 'cpu'
    '''

    def __init__(self, data: torch.Tensor, slices: Union[torch.Tensor, list, int], device: str = 'cpu') -> None:
        self._data = data

        if type(slices) is int:
            # each slice is a single type
            self._slices = torch.zeros((slices, 4), dtype=torch.long, device='cpu')
            for i in range(0, slices):
                self._slices[i][0] = i
                self._slices[i][1] = i
                self._slices[i][2] = i
                self._slices[i][3] = i + 1
            self._slices = self._slices.to(device)
        elif type(slices) is list:
            # 2d list, nx3
            self._slices = torch.as_tensor(slices, dtype=torch.long, device=device)
        else:
            self._slices = slices.to(device)
        # Don't backpropagate on slice tensors
        self._slices.requires_grad = False
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

    def tiling(self, tile_size: int, method: TilingMethod = TilingMethod.DEFAULT):
        assert tile_size > 0
        assert method == TilingMethod.DEFAULT, 'Only default tiling method is supported now.'
        slices = self._slices.tolist()
        subslices = []
        for slice in slices:
            index = slice[0]
            type = slice[1]
            start = slice[2]
            end = slice[3]
            for off in range(start, end, tile_size):
                subslices.append([index, type, off, min(off + tile_size, end)])
        return TensorSlice(self.data, subslices, self._slices.device)

    def schedule(self, op_name: str, *args, autotune: bool = False) -> CacheEntry:
        scheduler = schedulers[op_name]
        key = scheduler.get_key(*args)
<<<<<<< HEAD
<<<<<<< HEAD
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

        triton_op = getattr(triton_ops, op_name)
        for tile_size, tiling_method in product(scheduler.tile_sizes, scheduler.tiling_methods):
            input_tiles = self.tiling(tile_size, method=tiling_method)
            ms = do_bench(
                lambda: triton_op(
                    *args,
                    input_slices=self.slices,
                    input_tiles=input_tiles.slices,
                    tile_size=tile_size
                ),
                warmup=5,
                rep=10
            )
            if ms < best_ms:
                best_ms, best_op, best_config = ms, triton_op, BestConfig(tile_size=tile_size, input_tiles=input_tiles.slices)

=======
        if cache_entry := self._lookup_cache(op_name, key) is not None:
=======
        cache_entry = self._lookup_cache(op_name, key)

        if cache_entry is not None:
>>>>>>> 35dc19a (Update)
            return cache_entry.best_ms, cache_entry.best_config, cache_entry.best_op

        if autotune:
            best_ms, best_config, best_op = self.autotune(op_name, *args, scheduler=scheduler)
        else:
            best_ms, best_config, best_op = self.use_defaults(op_name, scheduler=scheduler)

        cache_entry = CacheEntry(best_ms, best_config, best_op)
        self._update_cache(op_name, key, cache_entry)
        return cache_entry

    def autotune(self, op_name: str, *args, scheduler) -> Tuple[float, BestConfig, callable]:
        best_op = getattr(torch_ops, op_name)
        best_ms = do_bench(lambda: best_op(*args, input_slices=self.slices), warmup=5, rep=10)
        best_config = BestConfig()

        triton_op = getattr(triton_ops, op_name)
        for tile_size, tiling_method in product(scheduler.tile_sizes, scheduler.tiling_methods):
            input_tiles = self.tiling(tile_size, method=tiling_method)
            ms = do_bench(
                lambda: triton_op(
                    *args,
                    input_slices=self.slices,
                    input_tiles=input_tiles.slices,
                    tile_size=tile_size
                ),
                warmup=5,
                rep=10
            )
            if ms < best_ms:
                best_ms, best_op, best_config = ms, triton_op, BestConfig(tile_size=tile_size, input_tiles=input_tiles.slices)

        return best_ms, best_config, best_op

    def use_defaults(self, op_name: str, scheduler: Scheduler) -> Tuple[float, BestConfig, callable]:
        input_tiles = self.tiling(scheduler.default_tile_size, method=scheduler.default_tiling_method)
        return 0.0, BestConfig(tile_size=scheduler.default_tile_size, input_tiles=input_tiles.slices), getattr(triton_ops, op_name)


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
            i, unique_types[i].item(), cur_index, cur_index + type_counts[i].item()])
        cur_index += type_counts[i].item()

    sorted_data = torch.index_select(data, dim=dim, index=type_indices) if not is_sorted else data
    return TensorSlice(sorted_data, type_list, device=device)
