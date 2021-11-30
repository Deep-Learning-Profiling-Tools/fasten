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
                         device=device, dtype=torch.float, requires_grad=True)
    input_tensor_slice = TensorSlice(input, input_slice)
    other_tensor_slice = TensorSlice(other, other_slice)

    # forward
    output = ops.bmm(input_tensor_slice, other_tensor_slice, backend=backend)
    truth = torch.tensor([[9, 12], [25, 32], [45, 56]],
                         device=device, dtype=torch.float)
    assert(torch.all(output == truth).item() is True)

    # backward
    output.backward(torch.ones_like(output))
    other_grad = torch.tensor(
        [[4, 4], [6, 6], [5, 5], [6, 6]], device=device, dtype=torch.float)
    assert(torch.all(other.grad == other_grad).item() is True)


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

        # warmup
        ops.bmm(input_tensor_slice, other_tensor_slice,
                backend=backend, nstreams=nstreams)

        # forward
        start_event.record()
        for _ in range(repeat):
            ret = ops.bmm(input_tensor_slice,
                          other_tensor_slice,  backend=backend, nstreams=nstreams)
        end_event.record()
        end_event.synchronize()

        ms = start_event.elapsed_time(end_event) / repeat

        print('{} forward: {} ms'.format(test_name, ms))

        # forward + backward
        start_event.record()
        other_tensor_slice.tensor.requires_grad = True
        for _ in range(repeat):
            ret = ops.bmm(input_tensor_slice, other_tensor_slice,
                          backend=backend, nstreams=nstreams)
            ret.backward(torch.ones_like(ret))
        end_event.record()
        end_event.synchronize()

        ms = start_event.elapsed_time(end_event) / repeat

        print('{} forward + backward: {} ms'.format(test_name, ms))

        return ret, other_tensor_slice.tensor.grad

    if backend is Backend.PYTHON:
        ret1, ret1_grad = run('Single stream', backend, nstreams=1)
        ret2, ret2_grad = run('Multi streams', backend, nstreams=8)
    else:
        ret1, ret1_grad = run('PYTHON backend', Backend.PYTHON)
        ret2, ret2_grad = run('NATIVE backend', Backend.NATIVE)

    assert(torch.allclose(ret1, ret2) is True)
    assert(torch.allclose(ret1_grad, ret2_grad) is True)


def test_correctness():
    correctness(Backend.PYTHON)
    correctness(Backend.NATIVE)


def test_speed():
    speed(Backend.PYTHON)
    speed(Backend.NATIVE)
