import torch
import torch.fx as fx
import torch_npu

class LayerNormModule(torch.nn.Module):
    def forward(self, input, weight, bias, epsilon):
        return torch.nn.functional.layer_norm(
            input, input.shape[-1:], weight, bias, epsilon
        )

def get_structure_replacements():
    return {
        "FusedLayerNorm": LayerNormModule,
    }
