import torch
from fasten import Ops as ops
from fasten import TensorSlice
from fasten import Backend

device = torch.device('cuda:0')


def correctness(backend: Backend):
    input_slice = [[1, 0, 2], [2, 2, 3]]
    input = torch.tensor([[1, 2], [3, 4], [5, 6]],
                         device=device, dtype=torch.float)
    other_slice = [[1, 0, 2], [2, 2, 4]]
    other = torch.tensor([[7, 8], [1, 2], [3, 4], [5, 6]],
                         device=device, dtype=torch.float)
    input_tensor_slice = TensorSlice(input, input_slice)
    other_tensor_slice = TensorSlice(other, other_slice)
    output = ops.bmm(input_tensor_slice, other_tensor_slice, backend=backend)
    truth = torch.tensor([[9, 12], [25, 32], [45, 56]],
                         device=device, dtype=torch.float)
    print(output)
    print(truth)
    assert(torch.all(output == truth).item() is True)


# 1. Compare single stream vs multiple streams
# 2. Compare bmm + index vs heterogenous bmm
def speed(backend: Backend):
    # 16 edge types
    input = torch.rand((16384, 16), dtype=torch.float, device=device)
    other = torch.rand((128, 16, 8), dtype=torch.float, device=device)
    input_types = torch.randint(
        0, 128, (16384,), dtype=torch.int, device=device)
    other_types = torch.arange(0, 128, device=device)

    def run(test_name, backend, nstreams=1):
        repeat = 3
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        input_tensor_slice = ops.compact(input, input_types)
        other_tensor_slice = ops.compact(other, other_types)

        ops.bmm(input_tensor_slice, other_tensor_slice,
                backend=backend, nstreams=nstreams)

        start_event.record()
        for _ in range(repeat):
            ret = ops.bmm(input_tensor_slice,
                          other_tensor_slice,  backend=backend, nstreams=nstreams)
        end_event.record()
        end_event.synchronize()

        ms = start_event.elapsed_time(end_event) / repeat

        print('{}: {} ms'.format(test_name, ms))

        return ret

    if backend is Backend.PYTHON:
        ret1 = run('Single stream', backend, nstreams=1)
        ret2 = run('Multi streams', backend, nstreams=8)
    else:
        ret1 = run('Default backend', backend)
        ret2 = run('{} backend'.format(backend), backend)

    assert(torch.allclose(ret1, ret2) is True)


def test_bmm_forward():
    correctness(Backend.PYTHON)
    correctness(Backend.NATIVE)
    speed(Backend.PYTHON)
    speed(Backend.NATIVE)


def test_bmm_backward():
    pass
