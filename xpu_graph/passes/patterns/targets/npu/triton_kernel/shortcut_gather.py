import torch
import torch_npu
import triton
import triton.language as tl
from typing import List

@triton.jit
def npu_shortcut_gather(in_ptr, out_ptr, PBLOCK: tl.constexpr, xnumel: tl.constexpr):
    offset = tl.program_id(0) * PBLOCK * xnumel
    idx = tl.arange(0, PBLOCK * xnumel)
    tmp = tl.load(in_ptr + offset + idx)
    tl.store(out_ptr + offset + idx,tmp)

from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta

def get_empty_result_tensor(input_tensor, dim, prefix_len):
    new_shape = list(input_tensor.shape)
    new_shape[dim] = prefix_len
    output_tensor = torch.empty(
        new_shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    return output_tensor

npu_def.define("shortcut_gather(Tensor input_tensor, int dim, int prefix_len) -> (Tensor)")
@impl(npu_lib, "shortcut_gather")
def shortcut_gather(
    input_tensor: torch.Tensor,
    dim: int,
    prefix_len: int,
) -> torch.Tensor:
    num_batch = input_tensor.shape[0]
    pblock = 4
    grid = ((num_batch - 1) // pblock + 1, 1, 1)

    output_tensor = get_empty_result_tensor(input_tensor, dim, prefix_len)

    xblock = output_tensor.numel() // num_batch

    if not (type(input_tensor) is torch._subclasses.fake_tensor.FakeTensor):
        npu_shortcut_gather[grid](
            input_tensor,
            output_tensor,
            PBLOCK = pblock,
            xnumel = xblock,
        )

    return output_tensor


# @shortcut_gather.register_fake
@impl(npu_meta, "shortcut_gather")
def shortcut_gather_fake(
    input_tensor: torch.Tensor,
    dim: int,
    prefix_len: int,
):
    return get_empty_result_tensor(input_tensor, dim, prefix_len)