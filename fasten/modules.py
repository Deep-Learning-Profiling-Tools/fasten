import warnings


def get_engine(module, engine_name: str):
    if engine_name is None:
        return module.TORCH
    if engine_name.lower() == 'magma':
        return module.MAGMA
    elif engine_name.lower() == 'torch':
        return module.TORCH
    elif engine_name.lower() == 'native':
        return module.NATIVE
    else:
        warnings.warn("Fasten: No such compute engine {}".format(
            engine_name), RuntimeError)


def get_module(use_cuda: bool):
    if use_cuda:
        import fasten_cuda as module
    else:
        import fasten_cpp as module
    return module
