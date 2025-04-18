import torch
import torch.fx as fx
import torch_npu

def ascend_compile(module: torch.nn.Module, example_inputs, config_dict, **kwargs) -> torch.nn.Module:
    torch.npu.set_compile_mode(jit_compile=False)

    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    if config_dict.get("mode", None) == "reduce-overhead":
        config.mode = config_dict["mode"]
    else:
        """
        TODO(zhangjihang): We have to use this, cause some case we have to use GE
        """
        config.experimental_config.keep_inference_input_mutations = True
        config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    compiled_module = npu_backend(module, example_inputs)

    return compiled_module