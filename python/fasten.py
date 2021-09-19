import torch


class EntrySlice:
    def __init__(self, type, range: slice) -> None:
        self._type = type
        self._range = range

    @property
    def type(self):
        return self._type

    @property
    def range(self):
        return self._range


def compact(entry_index: torch.Tensor, entry_type: torch.Tensor) -> tuple(torch.Tensor, list(EntrySlice)):
    '''
        Sort entry (node or edge) indices according to their types.

        :entry_index: the index of each entry
        :entry_type: the type of each entry
        :returns: <sorted entry_index tensor, entry_slice>
    '''
    sorted_entry_type, entry_type_indices = torch.sort(entry_type)
    sorted_entry_index = entry_index[entry_type_indices]
    unique_entry_type, entry_type_counts = torch.unique(
        sorted_entry_type, sorted=True, return_inverse=False, return_counts=True)

    entry_slices = []
    cur_index = 0
    for i in range(len(unique_entry_type)):
        # Ensure entry_slice is on CPU
        entry_slice = EntrySlice(
            unique_entry_type[i], (cur_index, cur_index + entry_type_counts[i]))
        cur_index += entry_type_counts[i]
        entry_slices.append(entry_slice)

    return sorted_entry_index, entry_slices


class Matmul:
    def __init__(self, native: bool = False, nstreams: int = 1) -> None:
        self._native = native
        self._nstreams = nstreams
        self._streams = []
        if nstreams == 1:
            self._streams.append(torch.cuda.current_stream())
        else:
            for _ in range(nstreams):
                self._streams.append(torch.cuda.Stream())

    def _apply_native(self, inputs: list(torch.Tensor), others: list(torch.Tensor)) -> list(torch.Tensor):
        pass

    def _apply_streams(self, inputs: list(torch.Tensor), others: list(torch.Tensor)) -> list(torch.Tensor):
        outputs = []
        for i in range(inputs):
            input = inputs[i]
            other = others[i]
            stream_id = i % self._nstreams
            with self._streams[stream_id]:
                outputs.append(torch.matmul(input, other))
        torch.cuda.synchronize()
        return outputs

    def apply(self, inputs: list(torch.Tensor), others: list(torch.Tensor)):
        if self._native is True:
            self._apply_native(inputs, others)
        else:
            self._apply_streams(inputs, others)
