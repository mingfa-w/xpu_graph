import math
import torch
import torch_mlu
import triton
import triton.language as tl
from typing import List

dtype_dict1 = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
}


@triton.jit
def mlu_triton_slice_sum_cat_kernel(
    output_ptr,
    input_ptr,
    indices_ptr,
    total_jobs,
    input_stride0,
    input_stride1,
    output_stride0,
    row,
    col,
    output_num,
    BLOCK_SIZE_T: tl.constexpr = 32,
    BLOCK_SIZE_I: tl.constexpr = 32,
    BLOCK_SIZE_R: tl.constexpr = 32,
    BLOCK_SIZE_C: tl.constexpr = 32,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    jobs_per_core = total_jobs // program_dim
    input_offset = input_ptr + program_id * jobs_per_core * input_stride0
    output_offset = output_ptr + program_id * jobs_per_core * output_stride0
    if program_id == program_dim - 1:
        jobs_per_core = total_jobs - (program_dim - 1) * jobs_per_core

    offset_i = tl.arange(0, BLOCK_SIZE_I)
    offset_c = tl.arange(0, BLOCK_SIZE_C)
    offset_r = tl.arange(0, BLOCK_SIZE_R)
    indices = tl.load(indices_ptr + offset_i)

    mask_c = offset_c < col
    mask_r = offset_r < row
    mask = mask_r[:, None] & mask_c[None, :]
    mask_cc = mask_c[None, :]
    input_data = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_R, BLOCK_SIZE_C), tl.float16)
    for idx in range(jobs_per_core):
        input_data[idx, :, :] = tl.load(
            input_offset + offset_r[:, None] * input_stride1 + offset_c[None, :],
            mask=mask,
        )
        input_offset += input_stride0

    # for idx in range(jobs_per_core):
    # value = input_data[idx,:,:]
    for i in range(output_num):
        start = indices[i * 2]
        end = indices[i * 2 + 1]
        mask_r = (offset_r < end) & (offset_r > start - 1)
        mask = mask_r[:, None] & mask_cc
        output_offset1 = output_offset
        for idx in range(jobs_per_core):
            value = input_data[idx, :, :]
            value_1 = tl.where(mask == 0, 0, value)
            # value_1 = value * mask
            sum_value = tl.sum(value_1, axis=0)
            tl.store(output_offset1 + i * col + offset_c, sum_value, mask=mask_c)
            output_offset1 += output_stride0


@torch.library.custom_op(
    "aten::mlu_triton_fuse_slice_sum_cat", mutates_args=(), device_types="mlu"
)
def mlu_triton_fuse_slice_sum_cat(
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
    grid = (processor_count, 1, 1)
    input_stride0 = input_tensor.stride(0)
    input_stride1 = input_tensor.stride(1)
    row = input_tensor.shape[1]
    col = input_tensor.shape[2]
    mini_block = 4
    total_jobs = input_tensor.shape[0]
    block_size_t = (
        (
            total_jobs
            - (processor_count - 1) * (total_jobs // processor_count)
            + mini_block
            - 1
        )
        // mini_block
    ) * mini_block
    block_size_i = ((output_num * 2 + mini_block - 1) // mini_block) * mini_block
    block_size_r = ((row + mini_block - 1) // mini_block) * mini_block
    block_size_c = ((col + mini_block - 1) // mini_block) * mini_block
    mlu_triton_slice_sum_cat_kernel[grid](
        output_tensor,
        input_tensor,
        slice_tensor,
        total_jobs,
        input_stride0,
        input_stride1,
        output_tensor.stride(0),
        row,
        col,
        output_num,
        block_size_t,
        block_size_i,
        block_size_r,
        block_size_c,
    )
    return output_tensor


@mlu_triton_fuse_slice_sum_cat.register_fake
def mlu_triton_fuse_slice_sum_cat_fake(
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
