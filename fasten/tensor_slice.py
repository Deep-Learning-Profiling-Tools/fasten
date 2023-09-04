from collections import OrderedDict
from typing import Tuple, Union

import torch
from triton.testing import do_bench

from .operators import torch_ops, triton_ops
from .scheduler import schedulers
from .utils import TilingMethod


class TensorSlice:
    '''
        Construct a TensorSlice data structure

        Args:
            tensor: The original type tensor, could be on either the CPU or the GPU.
                    It must have been sorted by types.
            slices: A 3-dim PyTorch Tensor, where each row represents [type_index, type, start, end].
                    It can also be a int or a list, then internally we transform it to a tensor.
            device: The device to put the slices on, default is 'cpu'
    '''

    def __init__(self, tensor: torch.Tensor, slices: Union[torch.Tensor, list, int], device: str = 'cpu') -> None:
        self._tensor = tensor

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

    def __len__(self):
        return self._slices.size(0)

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
    def tensor(self):
        return self._tensor

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

    def get_num_types(self) -> int:
        return self.__len__()

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
        return TensorSlice(self._tensor, subslices, self._slices.device)

    def schedule(self, op_name: str, *args, autotune: bool = False) -> Tuple[float, dict, callable]:
        scheduler = schedulers[op_name]
        key = scheduler.get_key(*args)
        if op_name in self._cache and key in self._cache[op_name]:
            return self._cache[op_name][key]
        if autotune:
            best_op = getattr(torch_ops, op_name)
            best_ms = do_bench(lambda: best_op(*args), warmup=5, rep=10)
            best_config = {'tile_size': None, 'input_tiles': None}
            triton_op = getattr(triton_ops, op_name)
            for tile_size in scheduler.tile_sizes:
                for tiling_method in scheduler.tiling_methods:
                    input_tiles = self.tiling(tile_size, method=tiling_method)
                    ms = do_bench(lambda: triton_op(*args, input_tiles=input_tiles.slices, tile_size=tile_size), warmup=5, rep=10)
                    if ms < best_ms:
                        best_ms = ms
                        best_op = triton_op
                        best_config = {'tile_size': tile_size, 'input_tiles': input_tiles.slices}
        else:
            input_tiles = self.tiling(scheduler.default_tile_size, method=scheduler.default_tiling_method)
            best_ms = 0.0  # not tuned
            best_op = getattr(triton_ops, op_name)
            best_config = {'tile_size': scheduler.default_tile_size, 'input_tiles': input_tiles.slices}
        return best_ms, best_config, best_op


def compact_tensor_types(tensor: torch.Tensor, types: torch.Tensor, type_dim: int = 0, descending: bool = False, device: str = 'cpu') -> Tuple[torch.Tensor, TensorSlice]:
    '''
        Sort a tensor (node or edge) according to their types.

        Args:
            tensor: The input tensor
            types: The type of each row
            type_dim: Which dimension of the tensor represents types
            descending: If true, sort the tensor in the descending order
            device: The device to put the slices. Note that tensor and sorted_types are still on the original device.

        Returns:
            tensor: A sorted tensor
            TensorSlice: The tensor's TensorSlice
            index: The original row indices in the sorted tensor
    '''
    # Must be stable sort
    sorted_types, type_indices = torch.sort(types, descending=descending, stable=True)
    sorted_tensor = torch.index_select(
        tensor, dim=type_dim, index=type_indices)
    # This function is different from torch.unique() in the sense that this function only eliminates
    # consecutive duplicate values. This semantics is similar to std::unique in C++.
    # torch.unique() may sort elements even if sorted is specified.
    unique_types, type_counts = torch.unique_consecutive(
        sorted_types, return_inverse=False, return_counts=True)

    types = []
    cur_index = 0
    for i in range(len(unique_types)):
        types.append([
            i, unique_types[i].item(), cur_index, cur_index + type_counts[i].item()])
        cur_index += type_counts[i].item()
    return sorted_tensor, TensorSlice(unique_types, types, device=device)
