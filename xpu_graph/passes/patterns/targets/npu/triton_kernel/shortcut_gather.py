import torch
import torch_npu
import triton
import triton.language as tl
from typing import List

@triton.jit
def npu_shortcut_gather_3D_dim1_prefix(in_ptr, out_ptr, pnumel: tl.constexpr, xnumel: tl.constexpr, rnumel: tl.constexpr, PBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    poffset = tl.program_id(0) * PBLOCK
    pidx = tl.arange(0, PBLOCK)[:,None,None] + poffset
    xidx = tl.arange(0, XBLOCK)[None,:,None]
    ridx = tl.arange(0, rnumel)[None,None,:]
    
    pmask = pidx<pnumel
    base_mask = tl.full((PBLOCK,XBLOCK,rnumel),True,tl.int1)
    mask = pmask&base_mask

    idx = pidx * xnumel * rnumel + xidx * rnumel + ridx

    tmp = tl.load(in_ptr + idx, mask)
    oidx = pidx * XBLOCK * rnumel + xidx * rnumel + ridx
    tl.store(out_ptr + oidx, tmp, mask)

@triton.jit
def npu_shortcut_gather_2D_dim1_prefix(in_ptr, out_ptr, pnumel: tl.constexpr, xnumel: tl.constexpr, PBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    poffset = tl.program_id(0) * PBLOCK
    pidx = tl.arange(0, PBLOCK)[:,None] + poffset
    xidx = tl.arange(0, XBLOCK)[None,:]
    
    pmask = pidx<pnumel
    base_mask = tl.full((PBLOCK,XBLOCK),True,tl.int1)
    mask = pmask&base_mask

    idx = pidx * xnumel + xidx

    tmp = tl.load(in_ptr + idx, mask)
    oidx = pidx * XBLOCK + xidx
    tl.store(out_ptr + oidx, tmp, mask)

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
    
    if not (dim == 1): 
        assert (0 and "continuous gather pattern not supported other dim yet")
    
    if not (type(input_tensor) is torch._subclasses.fake_tensor.FakeTensor):
        if len(input_tensor.shape) == 3:
            npu_shortcut_gather_3D_dim1_prefix[grid](
                input_tensor,
                output_tensor,
                pnumel = input_tensor.shape[0],
                xnumel = input_tensor.shape[1],
                rnumel = input_tensor.shape[2],
                PBLOCK = pblock,
                XBLOCK = prefix_len,
            )
        else:
            npu_shortcut_gather_2D_dim1_prefix[grid](
                input_tensor,
                output_tensor,
                pnumel = input_tensor.shape[0],
                xnumel = input_tensor.shape[1],
                PBLOCK = pblock,
                XBLOCK = prefix_len,
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