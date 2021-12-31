from enum import Enum
from typing import Optional, Union, Tuple
from collections import OrderedDict

import torch


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
            tensor: The original type tensor, could be on either the CPU or the GPU
            slices: A 3-dim PyTorch Tensor, where each row represents [type, start, end].
                    It can also be a int or a list, then internally we transform it to a tensor.
                    It must be on the CPU.
    '''

    def __init__(self, tensor: torch.tensor = None, slices: Union[torch.tensor, list, int] = None) -> None:
        self._tensor = tensor

        if type(slices) is int:
            self._slices = torch.zeros((slices, 3), dtype=torch.long)
            for i in range(0, slices):
                self._slices[i][0] = i
                self._slices[i][1] = i
                self._slices[i][2] = i + 1
        elif type(slices) is list:
            self._slices = torch.as_tensor(slices, dtype=torch.long)
        else:
            self._slices = slices.cpu()
        # Don't backpropagate on slice tensors
        self._slices.requires_grad = False

        # TODO(Keren): Simplify the internal data structures
        # Currently a map is reconstructed every time when it is passed to the C++ backend
        self._type_slice_dict = OrderedDict()
        for i in range(self._slices.size(0)):
            self._type_slice_dict[self._slices[i, 0].item()] = i

    def __len__(self):
        return self._slices.size(0)

    def __getitem__(self, key):
        return self._slices.__getitem__(key).item()

    def __setitem__(self, key, val):
        self._slices.__setitem__(key, val)

    def __eq__(self, __o: object) -> bool:
        return self._slices == __o

    def __contains__(self, key) -> bool:
        return key in self._type_slice_dict

    @property
    def start(self):
        return self._slices[0, 1].item()

    @property
    def stop(self):
        return self._slices[-1, 2].item()

    @property
    def slices(self):
        return self._slices

    @property
    def tensor(self):
        return self._tensor

    def types(self):
        return list(self._type_slice_dict.keys())

    def get_type(self, index: int) -> int:
        '''
            Get a type based on the index

            Args:
                index: The index of the slices

            Returns:
                type: type on the given index 
        '''
        return self._slices[index, 0].item()

    def get_slice(self, slice_type: int) -> slice:
        '''
            Get the slice of a specific type

            Args:
                slice_type: The type we want to get from the original slices

            Returns:
                slice: The slice of the given slice_type
        '''
        if slice_type not in self._type_slice_dict:
            raise RuntimeError(
                "TensorSlice does not have type {}".format(slice_type))
        row = self._type_slice_dict[slice_type]
        return slice(self._slices[row, 1].item(), self._slices[row, 2].item())

    def subslices(self, indices: slice):
        '''
            Construct subslices from the original slices

            Args:
                indices: The indices we want to extract from the original slices

            Returns:
                tensor_slice: A new TensorSlice object which has original offsets
        '''
        slices = self._slices[indices, :]
        return TensorSlice(self._tensor, slices)

    def extract(self, slice_types: list = None, relative: bool = True):
        '''
            Construct subslices of specific types from the original slices.

            Args:
                slice_types: The types we want to extract from the original slices. This array must be sorted.
                relative: If true, the new TensorSlice object which has relative offsets to the original tensor slice 

            Returns:
                tensor_slice: A new TensorSlice object
        '''
        if slice_types is None:
            slice_types = self.types()
        num_slice_types = len(slice_types)
        slices = torch.zeros((num_slice_types, 3), dtype=torch.long)
        offset = 0
        for i in range(num_slice_types):
            slice_type = slice_types[i]
            slices[i, 0] = slice_type
            if i == 0 and relative is True:
                offset = self._slices[i, 1].item()
            slices[i, 1] = self._slices[i, 1].item() - offset
            slices[i, 2] = self._slices[i, 2].item() - offset
        if self._tensor is None:
            tensor = None
        else:
            slice_start = slices[0, 1].item()+offset
            slice_stop = slices[-1, 2].item()+offset
            tensor = self._tensor[slice_start:slice_stop]
        return TensorSlice(tensor, slices)

    def expand(self, device: torch.device = torch.device('cpu')):
        '''
            Expand tensor slice to real tensors

            Args:
                device: On which device the new tensor should be

            Returns:
                tensor: A new tensor
        '''
        if self._tensor is not None:
            return self._tensor

        size = self.stop - self.start
        tensor = torch.zeros((size), device=device, dtype=torch.long)
        for i in range(len(self._slices)):
            type = self._slices[i, 0].item()
            start = self._slices[i, 1].item()
            end = self._slices[i, 2].item()
            tensor[start:end] = type
        return tensor


class TensorSliceTile:
    '''
        A Tiled iterator for TensorSlice
    '''

    def __init__(self, tensor_slice: TensorSlice, step: int = 1) -> None:
        self._tensor_slice = tensor_slice
        self._step = step
        self._cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._cur >= len(self._tensor_slice):
            raise StopIteration
        else:
            start = self._cur
            end = self._cur + self._step if self._cur + \
                self._step <= len(self._tensor_slice) else len(self._tensor_slice)
            self._cur += self._step
            return self._tensor_slice.subslices(slice(start, end))


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
    def compact(tensor: torch.tensor, types: torch.tensor, type_dim: int = 0, descending: bool = False) -> Tuple[torch.tensor, TensorSlice]:
        '''
            Sort a tensor (node or edge) according to their types.

            Args:
                tensor: A PyTorch tensor data
                types: The type of each row
                type_dim: Which dimension of the tensor represents types
                descending: If true, sort the tensor in the descending order

            Returns:
                tensor: A sorted tensor
                TensorSlice: The tensor's TensorSlice
                index: The original row indices in the sorted tensor 
        '''
        sorted_types, type_indices = torch.sort(types, descending=descending)
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
            # Ensure slice is on CPU
            types.append([
                unique_types[i].item(), cur_index, cur_index + type_counts[i].item()])
            cur_index += type_counts[i].item()
        return sorted_tensor, TensorSlice(sorted_types, types)

    @staticmethod
    def typed_sort(tensor: torch.tensor, tensor_slice: TensorSlice, dim: int = 0, base: Optional[Tuple[int]] = None, descending: bool = False) -> torch.tensor:
        '''
            XXX(Keren): Not tested yet
            Sort a tensor according to each dimension
        '''
        indices = tensor.select(dim, 0)

        if base is not None:
            for t in range(len(base)):
                indices = indices * base[t] + tensor.select(dim, t+1)

        base_max = torch.max(indices)
        indices = base_max * tensor_slice.tensor + indices
        _, sorted_indices = torch.sort(indices, descending=descending)
        sorted_tensor = torch.index_select(
            tensor, dim=dim, index=sorted_indices)

        return sorted_tensor

    @classmethod
    def _apply_streams(cls, op, input: torch.tensor, input_slices: TensorSlice, other: TensorSlice, other_slices: TensorSlice, output: torch.tensor, nstreams: int = 1) -> torch.tensor:
        '''
            Sort a tensor (node or edge) according to their types.

            Args:
                input: A PyTorch tensor
                input_slices: A TensorSlice
                other: A PyTorch tensor
                other_slices: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams

            Returns:
                A TensorSlice
        '''
        StreamPool.reserve(nstreams)

        for i in range(len(input_slices)):
            input_type = input_slices[i, 0]
            if input_type not in other_slices:
                continue
            input_slice = input_slices.get_slice(input_type)
            other_slice = other_slices.get_slice(input_type)
            input_tensor = input[input_slice, :]
            if len(input_tensor.size()) == cls.MAX_TENSOR_DIMS:
                input_tensor = torch.squeeze(other_tensor, 0)
            other_tensor = other[other_slice, :]
            if len(other_tensor.size()) == cls.MAX_TENSOR_DIMS:
                other_tensor = torch.squeeze(other_tensor, 0)
            stream_id = i % nstreams
            with torch.cuda.stream(StreamPool.get(stream_id)):
                output[input_slice, :] = op(input_tensor, other_tensor)

        if nstreams > 1:
            torch.cuda.synchronize()

        return output

    @classmethod
    def _validate_output(cls, input: torch.tensor, other: torch.tensor, output: torch.tensor) -> torch.tensor:
        '''
            Check if the given tensor shapes are valid and allocate an output tensor if needed

            Args:
                input: A PyTorch tensor
                other: A PyTorch tensor
                output: A PyTorch tensor

            Returns:
                A PyTorch tensor
        '''
        output_dims = []
        if len(input.size()) == cls.MIN_TENSOR_DIMS and len(other.size()) <= cls.MAX_TENSOR_DIMS:
            output_dims = [input.size()[0], other.size()[-1]]
        elif len(input.size()) == cls.MAX_TENSOR_DIMS and len(other.size()) <= cls.MAX_TENSOR_DIMS:
            output_dims = [input.size()[1], other.size()[-1]]
        else:
            raise RuntimeError("Fasten: do not support tensor shape input {} other {}".format(
                list(input.size()), list(other.size())))

        if output is None:
            output = torch.zeros((output_dims), device=input.device)

        if output_dims != list(output.size()):
            raise RuntimeError("Fasten: tensor shape error output {}".format(
                list(output.size())))

        return output

    @classmethod
    def bmm(cls, input: torch.tensor, input_slices: TensorSlice, other: torch.tensor, other_slices: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON, engine=None) -> torch.tensor:
        '''
            Batch matrix multiple input with other

            Args:
                input: A PyTorch tensor
                input_slices: A TensorSlice
                other: A PyTorch tensor
                other_slices: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend
                engine: If using the C++ backend, what algorithm to use

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBmm.apply(input, input_slices.slices, other, other_slices.slices, output, engine)
        else:
            return cls._apply_streams(torch.mm, input, input_slices, other, other_slices, output, nstreams)

    @classmethod
    def bmul(cls, input: torch.tensor, input_slices: TensorSlice, other: torch.tensor, other_slices: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON) -> torch.tensor:
        '''
            Batch multiple input with other

            Args:
                input: A PyTorch tensor
                input_slices: A TensorSlice
                other: A PyTorch tensor
                other_slices: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBmul.apply(input, input_slices.slices, other, other_slices.slices, output)
        else:
            return cls._apply_streams(torch.mul, input, input_slices, other, other_slices, output, nstreams)

    @classmethod
    def badd(cls, input: torch.tensor, input_slices: TensorSlice, other: torch.tensor, other_slices: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON) -> torch.tensor:
        '''
            Batch add input with other

            Args:
                input: A PyTorch tensor
                input_slices: A TensorSlice
                other: A PyTorch tensor
                other_slices: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBadd.apply(input, input_slices.slices, other, other_slices.slices, output)
        else:
            return cls._apply_streams(torch.add, input, input_slices, other, other_slices, output, nstreams)

    @classmethod
    def bsub(cls, input: torch.tensor, input_slices: TensorSlice, other: torch.tensor, other_slices: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON) -> torch.tensor:
        '''
            Batch sub input with other

            Args:
                input: A PyTorch tensor
                input_slices: A TensorSlice
                other: A PyTorch tensor
                other_slices: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBsub.apply(input, input_slices.slices, other, other_slices.slices, output)
        else:
            return cls._apply_streams(torch.sub, input, input_slices, other, other_slices, output, nstreams)

    def bdiv(cls, input: torch.tensor, input_slices: TensorSlice, other: torch.tensor, other_slices: TensorSlice, output: torch.tensor = None, nstreams: int = 1, backend: Backend = Backend.PYTHON) -> torch.tensor:
        '''
            Batch div input with other

            Args:
                input: A PyTorch tensor
                input_slices: A TensorSlice
                other: A PyTorch tensor
                other_slices: A TensorSlice
                output: A PyTorch tensor
                nstreams: Number of CUDA streams
                backend: Whether to use Python or Native C++ backend

            Returns:
                A PyTorch tensor
        '''
        output = cls._validate_output(input, other, output)
        if backend == Backend.NATIVE:
            return FastenBdiv.apply(input, input_slices.slices, other, other_slices.slices, output)
        else:
            return cls._apply_streams(torch.div, input, input_slices, other, other_slices, output, nstreams)
