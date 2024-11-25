import torch
import torch_mlu
import torch.fx as fx

def mlu_compile(module: torch.nn.Module, example_inputs) -> torch.nn.Module:
    from torch._inductor.compile_fx import compile_fx
    compiled_module = compile_fx(module, example_inputs)

    return compiled_module
