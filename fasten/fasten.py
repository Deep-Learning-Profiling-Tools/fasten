from enum import Enum
from typing import Union

import torch
import warnings

from .interfaces import *


class StreamPool:
    _streams = []
    if torch.cuda.is_available():
        _streams.append(torch.cuda.current_stream())

    @classmethod
    def add(cls, nstreams: int = 1) -> None:
        for _ in range(nstreams):
            cls._streams.append(torch.cuda.Stream())

    @classmethod
    def reserve(cls, nstreams: int = 1) -> None:
        if torch.cuda.is_available():
            if len(cls._streams) < nstreams:
                cls.add(nstreams - len(cls._streams))

    @classmethod
    def get(cls, stream_idx: int = 1) -> torch.cuda.Stream:
        if torch.cuda.is_available():
            return cls._streams[stream_idx]
        else:
            return None

    @classmethod
    def size(cls) -> int:
        return len(cls._streams)


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
        # Don't backpropagate on slice tensors
        self._slices.requires_grad = False

    @property
    def tensor(self):
        return self._tensor

    @property
    def slices(self):
        return self._slices


class Backend(Enum):
    '''
        The execution engines in fasten

        PYTHON: using python functions
        NATIVE: using c++ native functions
    '''
    PYTHON = 1
    NATIVE = 2


class Ops:
    MAX_TENSOR_DIMS = 3
    MIN_TENSOR_DIMS = 2

    '''
        This class contains all operations supported by fasten.

        The core idea of fasten is to support broadcastable batch operations for 2d matrices with various sizes.

        op(m1, m2)

        m1 and m2 can be either a 2d matrix or a 3d tensor. Based on m1 and m2's dims, there are four scenarios.

        1. m1-3d, m2-3d

        In this case, the outmost dimensions of m1 and m2 represent the type information. 
        So each sub-matrices in m1 or m2 should have the same shape.
        Since we only operate on submatrices with the same type, it falls back to the regular batch operation.

        2. m1-2d, m2-3d

        In this case, the sub-matrices in m1 can have different sizes, though their innermost dimensions are still the same.
        For example, m1 with shape [5, 4] can be composed of submatrices [3, 4] and [2, 4].

        3. m1-3d, m2-2d

        Similar to m2, the sub-matrices in m2 can have different sizes.

        4. m1-2d, m2-2d

        In this case, sub-matrices in m1 and m2 can have different sizes.

        MIN_TENSOR_DIMS: the minimum dimensions supported by fasten
        MAX_TENSOR_DIMS: the maximum dimensions supported by fasten
    '''

    @staticmethod
    def compact(tensor: torch.tensor, types: torch.tensor, descending: bool = False) -> TensorSlice:
        '''
            Sort a tensor (node or edge) according to their types.

            Args:
                tensor: the PyTorch tensor data
                types: the type of each entry
                descending: sort the tensor in the descending order

            Returns:
                A TensorSlice
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

    @classmethod
    def _apply_streams(cls, op, input: TensorSlice, other: TensorSlice, output: torch.tensor, nstreams: int = 1) -> torch.tensor:
        '''
            Sort a tensor (node or edge) according to their types.

            Args:
                input: A TensorSlice
                other: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams

            Returns:
                A TensorSlice
        '''
        StreamPool.reserve(nstreams)

        other_types = {}
        for i in range(len(other.slices)):
            other_types[other.slices[i][0].item()] = i

        for i in range(len(input.slices)):
            stream_id = i % nstreams
            input_slice = input.slices[i, :]
            input_type = input_slice[0].item()
            other_slice = other.slices[other_types[input_type], :]
            input_tensor = input.tensor[slice(
                input_slice[1].item(), input_slice[2].item()), :]
            if len(input_tensor.size()) == cls.MAX_TENSOR_DIMS:
                input_tensor = torch.squeeze(other_tensor, 0)
            other_tensor = other.tensor[slice(
                other_slice[1].item(), other_slice[2].item()), :]
            if len(other_tensor.size()) == cls.MAX_TENSOR_DIMS:
                other_tensor = torch.squeeze(other_tensor, 0)
            with torch.cuda.stream(StreamPool.get(stream_id)):
                output[slice(input_slice[1].item(), input_slice[2].item()), :] = op(
                    input_tensor, other_tensor)

        if nstreams > 1:
            torch.cuda.synchronize()

        return output

    @classmethod
    def _validate_output(cls, input: TensorSlice, other: TensorSlice, output: torch.tensor) -> torch.tensor:
        '''
            Check if the given tensor shapes are valid and allocate an output tensor if needed

            Args:
                input: A TensorSlice
                other: A TensorSlice
                output: A PyTorch tensor

            Returns:
                A PyTorch tensor
        '''
        output_dims = []
        if len(input.tensor.size()) == cls.MIN_TENSOR_DIMS and len(other.tensor.size()) <= cls.MAX_TENSOR_DIMS:
            output_dims = [input.tensor.size()[0], other.tensor.size()[-1]]
        elif len(input.tensor.size()) == cls.MAX_TENSOR_DIMS and len(other.tensor.size()) <= cls.MAX_TENSOR_DIMS:
            output_dims = [input.tensor.size()[1], other.tensor.size()[-1]]
        else:
            raise RuntimeError("Fasten: do not support tensor shape input {} other {}".format(
                list(input.size()), list(other.size())))

        if output is None:
            output = torch.zeros((output_dims), device=input.tensor.device)

        if output_dims != list(output.size()):
            raise RuntimeError("Fasten: tensor shape error output {}".format(
                list(output.size())))

        return output

    @classmethod
    def bmm(cls, input: TensorSlice, other: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON, engine=None) -> torch.tensor:
        '''
            Batch matrix multiple input with other

            Args:
                input: A TensorSlice
                other: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend
                engine: If using the C++ backend, what algorithm to use

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBmm.apply(input.tensor, input.slices, other.tensor, other.slices, output, engine)
        else:
            return cls._apply_streams(torch.mm, input, other, output, nstreams)

    @classmethod
    def bmul(cls, input: TensorSlice, other: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON) -> torch.tensor:
        '''
            Batch multiple input with other

            Args:
                input: A TensorSlice
                other: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBmul.apply(input.tensor, input.slices, other.tensor, other.slices, output)
        else:
            return cls._apply_streams(torch.mul, input, other, output, nstreams)

    @classmethod
    def badd(cls, input: TensorSlice, other: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON) -> torch.tensor:
        '''
            Batch add input with other

            Args:
                input: A TensorSlice
                other: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBadd.apply(input.tensor, input.slices, other.tensor, other.slices, output)
        else:
            return cls._apply_streams(torch.add, input, other, output, nstreams)

    @classmethod
    def bsub(cls, input: TensorSlice, other: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON) -> torch.tensor:
        '''
            Batch sub input with other

            Args:
                input: A TensorSlice
                other: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBsub.apply(input.tensor, input.slices, other.tensor, other.slices, output)
        else:
            return cls._apply_streams(torch.sub, input, other, output, nstreams)

    def bdiv(cls, input: TensorSlice, other: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON) -> torch.tensor:
        '''
            Batch div input with other

            Args:
                input: A TensorSlice
                other: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBdiv.apply(input.tensor, input.slices, other.tensor, other.slices, output)
        else:
            return cls._apply_streams(torch.div, input, other, output, nstreams)
