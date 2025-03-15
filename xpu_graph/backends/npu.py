from typing import Dict
import torch
import torch_npu
from .passes.npu.sum_bool import sum_bool_pass

def npu_compile(
    module: torch.nn.Module, inputs, config_dict: Dict
) -> torch.nn.Module:
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor import config
    from torch._inductor import list_mode_options
    # default means None
    mode = config_dict.get("mode", "default")
    dynamic = config_dict.get("dynamic", False)
    # torch bug: config values are assumed to be None.
    # Thus we do not use _TorchCompileInductorWrapper to pass custom passes
    # Refer to
    # [1] https://github.com/pytorch/pytorch/issues/139822
    # [2] https://github.com/pytorch/pytorch/pull/139833/files
    # from torch import _TorchCompileInductorWrapper
    # options = {}
    # inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    current_config = config.shallow_copy_dict()
    mode_dict = list_mode_options(mode, dynamic)
    current_config.update(mode_dict)
    current_config['post_grad_custom_post_pass'] = sum_bool_pass
    compiled_module = compile_fx(
        module, inputs, config_patches=current_config
    )

    return compiled_module