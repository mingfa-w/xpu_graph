import math
import torch
import torch_npu
import triton
import triton.language as tl
from typing import List

from xpu_graph.config import Multiflow

dtype_dict1 = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
}


@triton.jit
def npu_triton_slice_sum_cat_kernel(
    output_ptr,
    input_ptr,
    indices_ptr,
    input_stride0,
    input_stride1,
    output_stride0,
    row,
    col,
    output_num,
    FLOW_P_LEN: tl.constexpr, GRID_FLOW: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pbegin = tl.program_id(0) * FLOW_P_LEN
    pend = min(pbegin + FLOW_P_LEN, GRID_FLOW)
    for pid in range(pbegin, pend):
        program_id = pid
        indices_offset = indices_ptr
        input_offset = input_ptr + program_id * input_stride0
        output_offset = output_ptr + program_id * output_stride0

        index_row = tl.arange(0, BLOCK_SIZE_R)
        for coffs in range(0,col,BLOCK_SIZE_C):
            index_col = tl.arange(0, BLOCK_SIZE_C) + coffs
            mask_col = index_col < col
            for i in range(output_num):
                indices_start = tl.load(indices_offset + i*2)
                indices_end = tl.load(indices_offset + i*2 + 1)

                true_index_row = index_row[:,None] + indices_start

                slice_value = tl.load(
                    input_offset + true_index_row * input_stride1 + index_col[None,:],
                    mask = (true_index_row < indices_end) & mask_col,
                    other = 0
                )
                
                sum_value = tl.sum(slice_value, axis=0)
                tl.store(output_offset + i * col + index_col, sum_value, mask=mask_col)


from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta
npu_def.define("fuse_slice_sum_cat(Tensor input_tensor, Tensor slice_tensor, int processor_count, int output_num) -> (Tensor)")


@impl(npu_lib, "fuse_slice_sum_cat")
def fuse_slice_sum_cat(
    input_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    processor_count: int,
    output_num: int,
) -> torch.Tensor:
    
    output_tensor = torch.empty(
        [input_tensor.shape[0], output_num * input_tensor.shape[2]],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    input_stride0 = input_tensor.stride(0)
    input_stride1 = input_tensor.stride(1)
    batch = input_tensor.shape[0]
    row = input_tensor.shape[1]
    col = input_tensor.shape[2]
    total_jobs = batch
    UB_LIMIT=1392*16
    block_size_r = row
    block_size_c = min(max(UB_LIMIT // row, 1) , col)
    GRID_CNT = Multiflow.AivNum // Multiflow.FlowNum
    grid_flow = total_jobs
    flow_p_len = (grid_flow - 1) // GRID_CNT + 1
    grid = (GRID_CNT, 1, 1)
    
    
    npu_triton_slice_sum_cat_kernel[grid](
        output_tensor,
        input_tensor,
        slice_tensor,
        input_stride0,
        input_stride1,
        output_tensor.stride(0),
        row,
        col,
        output_num,
        flow_p_len, grid_flow,
        block_size_r,
        block_size_c,
    )
    return output_tensor



@impl(npu_meta, "fuse_slice_sum_cat")
def fuse_slice_sum_cat_fake(
    input_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    processor_count: int,
    output_num: int,
) -> torch.Tensor:
    output_tensor = torch.empty(
        [input_tensor.shape[0], output_num * input_tensor.shape[2]],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    return output_tensor
