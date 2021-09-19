import torch


class TensorSlice:
    def __init__(self, tensor: torch.Tensor, type_slices: list = []) -> None:
        self._tensor = tensor
        self._type_slices = type_slices

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

    def compact(self, tensor: torch.Tensor, types: torch.Tensor, descending: bool = False) -> TensorSlice:
        '''
            Sort a tensor (node or edge) according to their types.

            Args:
                tensor: the torch Tensor data
                types: the type of each entry
                descending: sort the tensor in the descending order

            Returns:
                TensorSlice
        '''
        sorted_type, type_indices = torch.sort(types, descending=descending)
        sorted_tensor = tensor[type_indices]
        unique_type, type_counts = torch.unique(
            sorted_type, sorted=True, return_inverse=False, return_counts=True)

        tensor_slice = TensorSlice(sorted_tensor)
        cur_index = 0
        for i in range(len(unique_type)):
            # Ensure slice is on CPU
            type_slice = (
                unique_type[i].item(), slice(cur_index, cur_index + type_counts[i].item()))
            tensor_slice.append(type_slice)
            cur_index += type_counts[i].item()

        return tensor_slice

    def bmm(self, input: TensorSlice, other: TensorSlice) -> torch.Tensor:
        '''
            Batch multiple input with other, where input and other contains many subslices with different sizes

            [b1, k] x [k, n]
            [b2, k] x [k, n]
            [b3, k] x [k, n]

            The output can be reshaped into a 2-d tensor using reshape(-1, n)

            Args:
                input: A TensorSlice
                other: A TensorSlice

            Returns:
                torch.Tensor: A 1-d torch Tensor
        '''

        def _apply_native(self, input: TensorSlice, other: TensorSlice) -> torch.Tensor:
            pass

        def _apply_streams(self, input: TensorSlice, other: TensorSlice) -> torch.Tensor:
            output_size: int = 0
            input_tensors = []
            other_tensors = []
            for i in range(len(input.type_slices)):
                input_slice = input.type_slices[i]
                other_slice = other.type_slices[i]
                input_tensor = input.tensor[input_slice[1], :]
                other_tensor = other.tensor[other_slice[1], :]
                input_tensors.append(input_tensor)
                other_tensors.append(other_tensor)
                output_size += input_tensor.shape[0] * other_tensor.shape[1]

            output = torch.Tensor(output_size, device=self._device)

            cur_size = 0
            for i in range(len(input.type_slices)):
                stream_id = i % self._nstreams
                input_tensor = input_tensors[i]
                other_tensor = other_tensors[i]
                size = input_tensor.shape[0] * other_tensor.shape[1]
                with self._streams[stream_id]:
                    torch.matmul(input_tensor, other_tensor,
                                 output[(cur_size, cur_size + size)])
                cur_size += size
            torch.cuda.synchronize()

            return output

        if self._native is True:
            self._apply_native(input, other)
        else:
            self._apply_streams(input, other)
