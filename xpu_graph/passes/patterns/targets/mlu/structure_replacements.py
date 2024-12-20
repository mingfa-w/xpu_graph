import torch
import torch.fx as fx
import torch_mlu
import torch_mlu_ops


class RMSNormModule(torch.nn.Module):
    def forward(self, inputs, weights, epsilon):
        return torch_mlu_ops.fused_rms_norm(
            inputs, None, weights, None, None, epsilon, False
        )


def get_structure_replacements():
    return {
        "FusedRMSNorm": RMSNormModule,
    }
