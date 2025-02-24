from typing import Dict
import torch
import torch_npu

def npu_compile(
    module: torch.nn.Module, inputs, config_dict: Dict
) -> torch.nn.Module:
    from torch._inductor.compile_fx import compile_fx
    from torch import _TorchCompileInductorWrapper

    mode = config_dict.get("mode", "default")
    options = {}
    dynamic = config_dict.get("dynamic", False)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    compiled_module = compile_fx(
        module, inputs, config_patches=inductor_backend.config
    )

    return compiled_module
