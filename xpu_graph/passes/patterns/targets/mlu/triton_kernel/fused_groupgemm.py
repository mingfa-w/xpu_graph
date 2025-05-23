import torch
import torch_mlu
from typing import List
from apex.contrib import grouped_gemm

@torch.library.custom_op("torch_mlu_triton::fused_grouped_gemm", mutates_args=())
def fused_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool
) -> torch.Tensor:
    output = grouped_gemm.ops.gmm(a, b, batch_sizes, trans_b=False)
    return output


@fused_grouped_gemm.register_fake
def fused_grouped_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool,
) -> torch.Tensor:
    output = torch.empty(a.shape[0], b.shape[-1], device=a.device, dtype=a.dtype)
    return output
