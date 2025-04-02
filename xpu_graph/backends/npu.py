<<<<<<< HEAD
from typing import Dict
import torch

def npu_compile(
    module: torch.nn.Module, inputs, config_dict: Dict
) -> torch.nn.Module:
    from torch._inductor.compile_fx import compile_fx
    from torch import _TorchCompileInductorWrapper

    # default means None. In torch, _TorchCompileInductorWrapper's apply_mode just passes.
    mode = config_dict.get("mode", "default")
    options = {}
    dynamic = config_dict.get("dynamic", False)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    compiled_module = compile_fx(
        module, inputs, config_patches=inductor_backend.config
    )

    return compiled_module
=======
import torch
import torch.fx as fx

def npu_compile(module: torch.nn.Module, example_inputs) -> torch.nn.Module:
    torch.npu.set_compile_mode(jit_compile=False)

    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    from torchair.configs.compiler_config import CompilerConfig

    """
    TODO(zhangjihang): We have to use this, cause some case we have to use GE
    """
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    config.experimental_config.frozen_parameter = True

    npu_backend = tng.get_npu_backend(compiler_config=config)
    compiled_module = npu_backend(module, example_inputs)

    return compiled_module
>>>>>>> c6888e1 (feat(ascend compiler): Enable xpu_graph + ge compiler for npu-910b)
