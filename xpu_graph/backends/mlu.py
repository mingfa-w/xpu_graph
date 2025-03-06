from typing import Dict
import torch
import torch_mlu


def mlu_compile(
    module: torch.nn.Module, example_inputs, config_dict: Dict
) -> torch.nn.Module:
    mode = config_dict.get("mode", "reduce-overhead")

    if mode == "cudagraphs":
        from torch._dynamo.backends.cudagraphs import cudagraphs

        return cudagraphs(module, example_inputs)

    from torch._inductor.compile_fx import compile_fx
    from torch import _TorchCompileInductorWrapper

    options = {}
    dynamic = config_dict.get("dynamic", True)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    compiled_module = compile_fx(
        module, example_inputs, config_patches=inductor_backend.config
    )

    return compiled_module
