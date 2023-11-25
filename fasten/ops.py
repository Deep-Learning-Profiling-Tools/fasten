import torch

from .tensor_slice import TensorSlice
from .utils import Engine, engine_ops


def execute_engine(*args, engine: Engine, tensor_slice: TensorSlice = None, op_name: str = None):
    if engine == Engine.AUTO or engine == Engine.TRITON:
        assert tensor_slice is not None, 'tensor_slice must be provided when using AUTO or TRITON engine'
        autotune = engine == Engine.AUTO
        cache_entry = tensor_slice.schedule(op_name, *args, autotune=autotune)
        best_config = cache_entry.best_config
        if best_config.input_tiles is None:
            return cache_entry.best_op(*args, input_slices=tensor_slice.slices)
        else:
            return cache_entry.best_op(*args, input_slices=tensor_slice.slices, **(best_config.asdict()))
    elif engine == Engine.TORCH:
        engine_op = getattr(engine_ops[engine], op_name)
        return engine_op(*args, input_slices=tensor_slice.slices)
    else:
        raise NotImplementedError(f'Engine {engine} is not implemented')


class FastenSegmentMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, other: torch.Tensor,
                tensor_slice: TensorSlice, engine: Engine = Engine.AUTO):
        ctx.save_for_backward(input, other)
        ctx.engine = engine
        ctx.tensor_slice = tensor_slice
        return execute_engine(input, other, engine=engine, tensor_slice=tensor_slice, op_name='segment_matmul_forward')

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, other = ctx.saved_tensors
        grad_input = execute_engine(
            grad_output, other,
            engine=ctx.engine, tensor_slice=ctx.tensor_slice, op_name='segment_matmul_backward_input')
        grad_other = execute_engine(
            input, grad_output, other,
            engine=ctx.engine, tensor_slice=ctx.tensor_slice, op_name='segment_matmul_backward_other')
        return grad_input, grad_other, None, None


fasten_segment_matmul = FastenSegmentMatmul.apply
