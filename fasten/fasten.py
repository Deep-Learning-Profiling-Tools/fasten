from enum import Enum
from typing import Union

import torch
import bmm
import warnings


class TensorSlice:
    '''
        Construct a TensorSlice data structure

        Args:
            tensor: A PyTorch Tensor
            slices: A 3-dim PyTorch Tensor, where each row represents [type, start, end].
                    It can also be a list, then internally we transform it to a tensor
    '''

    def __init__(self, tensor: torch.tensor, slices: Union[torch.tensor, list] = None) -> None:
        self._tensor = tensor
        if type(slices) is list:
            self._slices = torch.as_tensor(slices)
        else:
            self._slices = slices

    @property
    def tensor(self):
        return self._tensor

    @property
    def slices(self):
        return self._slices


class Backend(Enum):
    '''
        The execution engines in fasten

        TORCH: using torch compute functions
        MAGMA: using magma's functions
        NATIVE: using fasten's native c++ interface
    '''
    PYTHON = 1
    NATIVE = 2


class HeteroOps:
    '''
        Operations for heterogenous graphs

        Args:
            device: execution device
            backend: math operation execution engine
            nstreams: the number of used streams
    '''

    def __init__(self, device: torch.device, backend: Backend = Backend.PYTHON, nstreams: int = 1) -> None:
        self._device = device
        self._backend = backend
        self._nstreams = nstreams
        self._streams = []
        if 'cuda' in self._device:
            if nstreams == 1:
                self._streams.append(torch.cuda.current_stream())
            else:
                if backend is Backend.PYTHON:
                    for _ in range(nstreams):
                        self._streams.append(torch.cuda.Stream())
                else:
                    warnings.warn(
                        'Fasten bmm only supports multi-streaming for the PYTHON backend', RuntimeWarning)

    def compact(self, tensor: torch.tensor, types: torch.tensor, descending: bool = False) -> TensorSlice:
        '''
            Sort a tensor (node or edge) according to their types.

            Args:
                tensor: the torch tensor data
                types: the type of each entry
                descending: sort the tensor in the descending order

            Returns:
                TensorSlice
        '''
        sorted_types, type_indices = torch.sort(types, descending=descending)
        sorted_tensor = tensor[type_indices]
        # This function is different from torch.unique() in the sense that this function only eliminates
        # consecutive duplicate values. This semantics is similar to std::unique in C++.
        # torch.unique() may sort elements even if sorted is specified.
        unique_types, type_counts = torch.unique_consecutive(
            sorted_types, return_inverse=False, return_counts=True)

        types = []
        cur_index = 0
        for i in range(len(unique_types)):
            # Ensure slice is on CPU
            types.append([
                unique_types[i].item(), cur_index, cur_index + type_counts[i].item()])
            cur_index += type_counts[i].item()

        return TensorSlice(sorted_tensor, torch.as_tensor(types))

    def bmm(self, input: TensorSlice, other: TensorSlice, engine = None) -> torch.tensor:
        '''
            Batch multiple input with other, where input and other contains many subslices with different sizes

            [m1, k] x [k, n] -> [m1, n]
            [m2, k] x [k, n] -> [m2, n]
            [m3, k] x [k, n] -> [m3, n]

            The output can be reshaped into a 2-d tensor using reshape(-1, n)

            Args:
                input: A TensorSlice
                other: A TensorSlice

            Returns:
                torch.tensor: A 1-d torch Tensor
        '''

        def apply_native(input: TensorSlice, other: TensorSlice, engine) -> torch.tensor:
            return bmm.FastenBmm(input.tensor, input.slices, other.tensor, other.slices, engine)

        def apply_streams(input: TensorSlice, other: TensorSlice) -> torch.tensor:
            output_size = input.tensor.shape[0] * other.tensor.shape[-1]
            output = torch.empty(
                output_size, device=self._device, dtype=input.tensor.dtype)

            other_types = {}
            for i in range(len(other.type_slices)):
                other_types[other.type_slices[i][0]] = i

            cur_size = 0
            for i in range(len(input.type_slices)):
                stream_id = i % self._nstreams
                input_slice = input.slices[i]
                input_type = input_slice[0]
                other_slice = other.slices[other_types[input_type]]
                input_tensor = input.tensor[slice(
                    input_slice[1].item(), input_slice[2].item()), :]
                other_tensor = other.tensor[slice(
                    other_slice[1].item(), input_slice[2].item()), :].squeeze(0)
                size = input_tensor.shape[0] * other_tensor.shape[-1]
                output_slice = output[slice(cur_size, cur_size + size)]
                output_slice = output_slice.view(
                    input_tensor.shape[0], other_tensor.shape[-1])
                with torch.cuda.stream(self._streams[stream_id]):
                    torch.mm(input_tensor, other_tensor, out=output_slice)
                cur_size += size

            if self._nstreams >= 1:
                torch.cuda.synchronize()

            return output.view(input.tensor.shape[0], other.tensor.shape[-1])

        if self._backend == Backend.NATIVE:
            return apply_native(input, other, engine)
        else:
            return apply_streams(input, other)
