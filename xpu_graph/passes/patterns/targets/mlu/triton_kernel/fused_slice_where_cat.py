import math
import torch
import torch_mlu
import triton
import triton.language as tl

@triton.jit
def mlu_triton_slice_where_cat_kernel(
    output_ptr,
    where_ptr,
    slice_ptr,
    slice_param_ptr,
    where_stride0,
    slice_stride0,
    output_stride0,
    total_jobs,
    where_col,
    slice_col,
    slice_min,
    slice_max,
    batch,
    slice_num,
    BLOCK_SIZE_I: tl.constexpr = 32,
    BLOCK_SIZE_S: tl.constexpr = 32,
    BLOCK_SIZE_D: tl.constexpr = 32,
    BLOCK_SIZE_WHERE_C: tl.constexpr = 32,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)

    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)

    block_jobs = total_jobs // program_dim
    remainder_jobs = total_jobs % program_dim
    block_jobs_r = block_jobs + (1 if program_id < remainder_jobs else 0)
    start_index = program_id * block_jobs + min(program_id, remainder_jobs)

    param_offset = slice_param_ptr

    offset_param = tl.arange(0, BLOCK_SIZE_I)
    offset_slice = tl.arange(0, BLOCK_SIZE_S)
    offset_where = tl.arange(0, 1)
    offset_output = tl.arange(0, BLOCK_SIZE_S)

    mask_param = offset_param < slice_num * 2
    mask_where = offset_where < where_col

    slice_param = tl.load(param_offset + offset_param, mask=mask_param)

    for block_idx in range(block_jobs_r):
        slice_offset = slice_ptr + (start_index + block_idx) * slice_stride0
        where_offset = where_ptr + (start_index + block_idx) * where_stride0
        output_offset = output_ptr + (start_index + block_idx) * output_stride0

        where_data = tl.load(where_offset + offset_where, mask=mask_where)
        where_expand = tl.broadcast_to(where_data, BLOCK_SIZE_S)

        slice_start = slice_param[0::2]
        slice_end = slice_param[1::2]
        slice_diff = slice_end - slice_start

        for i in range(slice_num):
            mask_slice = offset_slice < slice_diff[i]
            mask_output = offset_output < slice_diff[i]
            slice_data = tl.load(slice_offset + offset_slice + slice_start[i], mask=mask_slice)
            #where_expand = tl.broadcast_to(where_data, BLOCK_SIZE_S)
            slice_where = tl.where(where_expand, 0, slice_data)
            tl.store(output_offset + i * slice_diff[i] + offset_output, slice_where, mask=mask_output)


@torch.library.custom_op(
    "torch_mlu_triton::fuse_slice_where_cat", mutates_args=(), device_types="mlu"
)
def fuse_slice_where_cat(
    where_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    slice_param: torch.Tensor,
    slice_min: int,
    slice_max: int,
    zeros_param_dim1: int,
    processor_count: int,
    slice_num: int,
) -> torch.Tensor:
    output_tensor = torch.empty(
        [slice_tensor.shape[0], slice_num * zeros_param_dim1],
        dtype=slice_tensor.dtype,
        device=slice_tensor.device,
    )
    where_stride0 = where_tensor.stride(0)
    where_stride1 = where_tensor.stride(1)

    slice_stride0 = slice_tensor.stride(0)
    slice_stride1 = slice_tensor.stride(1)

    output_stride0 = output_tensor.stride(0)
    output_stride1 = output_tensor.stride(1)

    batch = slice_tensor.shape[0]
    where_col = where_tensor.shape[1]
    slice_col = slice_max - slice_min
    total_jobs = batch

    mini_block = 4
    block_size_i = ((slice_num * 2 + mini_block - 1) // mini_block) * mini_block
    block_size_slice_max = ((slice_max + mini_block - 1) // mini_block) * mini_block
    block_size_slice_diff = ((slice_col + mini_block - 1) // mini_block) * mini_block
    block_size_where_c = ((where_col + mini_block - 1) // mini_block) * mini_block
    grid = (processor_count, 1, 1)

    mlu_triton_slice_where_cat_kernel[grid](
        output_tensor,
        where_tensor,
        slice_tensor,
        slice_param,
        where_stride0,
        slice_stride0,
        output_stride0,
        batch,
        where_col,
        slice_col,
        slice_min,
        slice_max,
        total_jobs,
        slice_num,
        block_size_i,
        block_size_slice_max,
        block_size_slice_diff,
        block_size_where_c,
    )
    return output_tensor

@fuse_slice_where_cat.register_fake
def fuse_slice_where_cat_fake(
    where_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    slice_param: torch.Tensor,
    slice_min: int,
    slice_max: int,
    zeros_param_dim1: int,
    processor_count: int,
    slice_num: int,
) -> torch.Tensor:
    output_tensor = torch.empty(
        [slice_tensor.shape[0], slice_num * zeros_param_dim1],
        dtype=slice_tensor.dtype,
        device=slice_tensor.device,
    )
    return output_tensor
