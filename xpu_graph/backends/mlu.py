from typing import Dict
import torch
import torch_mlu


def mlu_compile(
    module: torch.nn.Module, example_inputs, config_dict: Dict
) -> torch.nn.Module:
    from torch._inductor.compile_fx import compile_fx
    from torch import _TorchCompileInductorWrapper

    mode = config_dict.get("mode", "reduce-overhead")
    options = {}
    dynamic = config_dict.get("dynamic", True)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    compiled_module = compile_fx(
        module, example_inputs, config_patches=inductor_backend.config
    )

    return compiled_module
