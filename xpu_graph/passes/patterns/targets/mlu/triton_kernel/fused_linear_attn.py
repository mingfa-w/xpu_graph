import math
import torch
import torch_mlu
import triton
import triton.language as tl
from typing import List
from xpu_graph.utils import logger

from .linear_attention_kernel import attention


@torch.library.custom_op(
    "torch_mlu_triton::linear_attn", mutates_args=(), device_types="mlu"
)
def linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    causal: bool,
    sm_scale: float,
    has_bias: bool,
) -> torch.Tensor:
    expand = False
    if has_bias:
        if len(bias.shape) == 2:
            expand = False
        elif len(bias.shape) == 3 and bias.shape[0] == 1:
            bias = bias.squeeze(0)
            expand = False
        elif len(bias.shape) == 4:
            expand = True
        else:
            logger.error(f"Linear Atention: unexpected shape: {bias.shape}")
    output_tensor = attention(q, k, v, bias, causal, has_bias, expand, 1)
    return output_tensor


@linear_attn.register_fake
def linear_attn_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    causal: bool,
    sm_scale: float,
    has_bias: bool,
) -> torch.Tensor:
    return torch.empty_like(q)
