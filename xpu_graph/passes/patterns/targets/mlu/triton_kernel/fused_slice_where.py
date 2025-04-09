import torch
import torch_mlu
import triton
import triton.language as tl
from typing import List
from . import libentry


@libentry.libentry()
@triton.jit
def mlu_triton_slice_where_kernel(
    output_ptr,
    where_ptr,
    slice_ptr,
    start_indices_ptr,
    where_stride,
    slice_stride,
    slice_row,
    slice_len,
    slice_num,
    loop,
    total_jobs,
    BLOCK_SIZE_R: tl.constexpr = 32,
    BLOCK_SIZE_C: tl.constexpr = 32,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)

    block_jobs = total_jobs // program_dim
    remainder_jobs = total_jobs % program_dim
    block_jobs_r = block_jobs + (1 if program_id < remainder_jobs else 0)
    block_start_index = program_id * block_jobs + min(program_id, remainder_jobs)

    for loop_ in range(loop):
        for block_idx in range(block_jobs_r):
            slice_idx = block_idx + block_start_index
            start_index = tl.load(start_indices_ptr + slice_idx)

            slice_block_ptr = tl.make_block_ptr(
                base=slice_ptr,
                shape=(slice_row, slice_len),
                strides=(slice_stride, 1),
                offsets=(loop_ * BLOCK_SIZE_R, start_index),
                block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
                order=(1, 0),
            )
            slice_data = tl.load(slice_block_ptr, boundary_check=(0,), padding_option=0)

            where_block_ptr = tl.make_block_ptr(
                base=where_ptr,
                shape=(slice_row, 1),
                strides=(where_stride, 1),
                offsets=(loop_ * BLOCK_SIZE_R, 0),
                block_shape=(BLOCK_SIZE_R, 1),
                order=(1, 0),
            )
            where_data = tl.load(where_block_ptr, boundary_check=(0,), padding_option=0)

            slice_where = tl.where(where_data, 0, slice_data)

            output_ptr_offset = output_ptr + (slice_idx * slice_row) * slice_len
            output_block_ptr = tl.make_block_ptr(
                base=output_ptr_offset,
                shape=(slice_row, slice_len),
                strides=(slice_len, 1),
                offsets=(loop_ * BLOCK_SIZE_R, 0),
                block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
                order=(1, 0),
            )
            tl.store(output_block_ptr, slice_where, boundary_check=(0,))


@torch.library.custom_op("torch_mlu_triton::fuse_slice_where", mutates_args=(), device_types="mlu")
def fuse_slice_where(
    where_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    start_indices: torch.Tensor,
    slice_len: int,
    slice_num: int,
) -> torch.Tensor:
    output_tensors = torch.empty(
        (slice_num * slice_tensor.shape[0], slice_len),
        device=slice_tensor.device,
        dtype=slice_tensor.dtype,
    )
    where_stride = where_tensor.stride(0)
    slice_stride = slice_tensor.stride(0)

    slice_row = slice_tensor.shape[0]

    block_r = slice_row
    block_c = slice_len

    size_of_dtype = 2
    if slice_tensor.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = 416 * 1024
    if (block_r * block_c * size_of_dtype) * 3 > nram_limit:
        block_r = nram_limit // 3 // size_of_dtype // block_c
    
    loop = (slice_row + block_r - 1) // block_r
    total_jobs = slice_num

    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    grid = (processor_count, 1, 1)
    mlu_triton_slice_where_kernel[grid](
        output_tensors,
        where_tensor,
        slice_tensor,
        start_indices,
        where_stride,
        slice_stride,
        slice_row,
        slice_len,
        slice_num,
        loop,
        total_jobs,
        block_r,
        block_c,
    )
    return output_tensors


@fuse_slice_where.register_fake
def fuse_slice_where_fake(
    where_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    start_indices: torch.Tensor,
    slice_len: int,
    slice_num: int,
) -> torch.Tensor:
    output_tensors = torch.empty(
        (slice_num * slice_tensor.shape[0], slice_len),
        device=slice_tensor.device,
        dtype=slice_tensor.dtype,
    )
    return output_tensors
