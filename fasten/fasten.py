import torch


class TensorSlice:
    def __init__(self, tensor: torch.tensor, type_slices: list = None) -> None:
        self._tensor = tensor
        self._type_slices = type_slices or []

    @property
    def tensor(self):
        return self._tensor

    @property
    def type_slices(self):
        return self._type_slices

    def append(self, type_slice):
        self._type_slices.append(type_slice)


class HeteroOps:
    def __init__(self, device: torch.device, native: bool = False, nstreams: int = 1) -> None:
        self._device = device
        self._native = native
        self._nstreams = nstreams
        self._streams = []
        if nstreams == 1:
            self._streams.append(torch.cuda.current_stream())
        else:
            for _ in range(nstreams):
                self._streams.append(torch.cuda.Stream())

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

        tensor_slice = TensorSlice(sorted_tensor)
        cur_index = 0
        for i in range(len(unique_types)):
            # Ensure slice is on CPU
            type_slice = (
                unique_types[i].item(), slice(cur_index, cur_index + type_counts[i].item()))
            tensor_slice.append(type_slice)
            cur_index += type_counts[i].item()

        return tensor_slice

    def bmm(self, input: TensorSlice, other: TensorSlice) -> torch.tensor:
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

        def apply_native(input: TensorSlice, other: TensorSlice) -> torch.tensor:
            pass

        def apply_streams(input: TensorSlice, other: TensorSlice) -> torch.tensor:
            output_size = input.tensor.shape[0] * other.tensor.shape[-1]
            output = torch.empty(
                output_size, device=self._device, dtype=input.tensor.dtype)

            cur_size = 0
            for i in range(len(input.type_slices)):
                stream_id = i % self._nstreams
                input_slice = input.type_slices[i]
                other_slice = other.type_slices[i]
                input_tensor = input.tensor[input_slice[1], :]
                other_tensor = other.tensor[other_slice[1], :].squeeze(0)
                size = input_tensor.shape[0] * other_tensor.shape[-1]
                output_slice = output[slice(cur_size, cur_size + size)]
                output_slice = output_slice.view(
                    input_tensor.shape[0], other_tensor.shape[-1])
                with torch.cuda.stream(self._streams[stream_id]):
                    torch.matmul(input_tensor, other_tensor, out=output_slice)
                cur_size += size

            if self._nstreams >= 1:
                torch.cuda.synchronize()

            return output.view(input.tensor.shape[0], other.tensor.shape[-1])

        if self._native is True:
            return apply_native(input, other)
        else:
            return apply_streams(input, other)
