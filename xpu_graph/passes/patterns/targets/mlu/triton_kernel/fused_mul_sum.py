import torch
import torch_mlu
import triton
import triton.language as tl
from typing import List, Optional

@triton.jit
def mlu_triton_slice_mul_sum_kernel(
    output_ptr,
    output_stride,
    mul0_ptr,
    mul1_ptr,
    mul0_stride0,
    mul0_stride1,
    mul1_stride0,
    mul1_stride1,
    batch,
    input_row,
    total_jobs,
    loop,
    slice_l,
    is_mul1_multi_dim: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr = 16,
    BLOCK_SIZE_R: tl.constexpr = 16,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)

    block_jobs = total_jobs // program_dim
    remainder_jobs = total_jobs % program_dim
    block_jobs_r = block_jobs + (1 if program_id < remainder_jobs else 0)
    block_start_index = program_id * block_jobs + min(program_id, remainder_jobs)

    for loop_ in range(loop):
        mul0_block_ptr = tl.make_block_ptr(
            base=mul0_ptr,
            shape=(batch, input_row, 1),
            strides=(mul0_stride0, mul0_stride1, 1),
            offsets=(loop_ * BLOCK_SIZE_B + block_start_index, 0, slice_l),
            block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_R, 1),
            order=(2, 1, 0),
        )
        mul0_data = tl.load(mul0_block_ptr, boundary_check=(0,), padding_option=0)
        if is_mul1_multi_dim:
            mul1_block_ptr = tl.make_block_ptr(
                base=mul1_ptr,
                shape=(batch, input_row, 1),
                strides=(mul1_stride0, mul1_stride1, 1),
                offsets=(loop_ * BLOCK_SIZE_B + block_start_index, 0, 0),
                block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_R, 1),
                order=(2, 1, 0),
            )
        else:
            mul1_block_ptr = tl.make_block_ptr(
                base=mul1_ptr,
                shape=(1, input_row, 1),
                strides=(mul1_stride0, 1, 1),
                offsets=(0, 0, 0),
                block_shape=(1, BLOCK_SIZE_R, 1),
                order=(2, 1, 0),
            )
        mul1_data = tl.load(mul1_block_ptr, boundary_check=(0,), padding_option=0)
        mul_result = mul0_data * mul1_data
        sum_result = tl.sum(mul_result, axis=1)
        output_block_ptr = tl.make_block_ptr(
            base=output_ptr,
            shape=(batch, 1),
            strides=(output_stride, 1),
            offsets=(loop_ * BLOCK_SIZE_B + block_start_index, 0),
            block_shape=(BLOCK_SIZE_B, 1),
            order=(1, 0),
        )
        tl.store(output_block_ptr, sum_result, boundary_check=(0,))

@triton.jit
def mlu_triton_mul_sum_kernel(
    output_ptr,
    output_stride,
    mul0_ptr,
    mul1_ptr,
    mul0_stride,
    mul1_stride,
    batch,
    input_row,
    total_jobs,
    loop,
    is_mul1_multi_dim: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr = 16,
    BLOCK_SIZE_R: tl.constexpr = 16,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)

    block_jobs = total_jobs // program_dim
    remainder_jobs = total_jobs % program_dim
    block_jobs_r = block_jobs + (1 if program_id < remainder_jobs else 0)
    block_start_index = program_id * block_jobs + min(program_id, remainder_jobs)

    for loop_ in range(loop):
        mul0_block_ptr = tl.make_block_ptr(
            base=mul0_ptr,
            shape=(batch, input_row, 1),
            strides=(mul0_stride, 1, 1),
            offsets=(loop_ * BLOCK_SIZE_B + block_start_index, 0, 0),
            block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_R, 1),
            order=(2, 1, 0),
        )
        mul0_data = tl.load(mul0_block_ptr, boundary_check=(0,), padding_option=0)
        if is_mul1_multi_dim:
            mul1_block_ptr = tl.make_block_ptr(
                base=mul1_ptr,
                shape=(batch, input_row, 1),
                strides=(mul1_stride, 1, 1),
                offsets=(loop_ * BLOCK_SIZE_B + block_start_index, 0, 0),
                block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_R, 1),
                order=(2, 1, 0),
            )
        else:
            mul1_block_ptr = tl.make_block_ptr(
                base=mul1_ptr,
                shape=(1, input_row, 1),
                strides=(mul1_stride, 1, 1),
                offsets=(0, 0, 0),
                block_shape=(1, BLOCK_SIZE_R, 1),
                order=(2, 1, 0),
            )
        mul1_data = tl.load(mul1_block_ptr, boundary_check=(0,), padding_option=0)
        mul_result = mul0_data * mul1_data
        sum_result = tl.sum(mul_result, axis=1)
        output_block_ptr = tl.make_block_ptr(
            base=output_ptr,
            shape=(batch, 1),
            strides=(output_stride, 1),
            offsets=(loop_ * BLOCK_SIZE_B + block_start_index, 0),
            block_shape=(BLOCK_SIZE_B, 1),
            order=(1, 0),
        )
        tl.store(output_block_ptr, sum_result, boundary_check=(0,))

@torch.library.custom_op("torch_mlu_triton::fused_mul_sum", mutates_args=())
def fused_mul_sum(
    mul0: torch.Tensor,
    mul1: torch.Tensor,
    slice_l: Optional[int],
) -> torch.Tensor:
    batch = mul0.shape[0]
    input_row = mul0.shape[1]
    block_size_b0 = batch
    block_size_b1 = mul1.shape[0]
    block_size_r = input_row

    is_mul1_multi_dim = False
    if block_size_b1 > 1:
        is_mul1_multi_dim = True

    size_of_dtype = 2
    if mul0.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = 416 * 1024
    if block_size_b0 * block_size_r * size_of_dtype * 2 + \
        block_size_b1 * block_size_r * size_of_dtype + \
        block_size_b0 * size_of_dtype > nram_limit: \
        block_size_b0 = (nram_limit // size_of_dtype - block_size_b1 * block_size_r) \
                        // (block_size_r * 2 + 1)

    output_tensor = torch.zeros(
        (batch, 1),
        device=mul0.device,
        dtype=mul0.dtype,
    )

    loop = (batch + block_size_b0 - 1) // block_size_b0
    total_jobs = block_size_b0
    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    grid = (processor_count, 1, 1)

    if not slice_l:
        mlu_triton_mul_sum_kernel[grid](
            output_tensor,
            output_tensor.stride(0),
            mul0,
            mul1,
            mul0.stride(0),
            mul1.stride(0),
            batch,
            input_row,
            total_jobs,
            loop,
            is_mul1_multi_dim,
            block_size_b0,
            block_size_r,
        )
    else:
        mlu_triton_slice_mul_sum_kernel[grid](
            output_tensor,
            output_tensor.stride(0),
            mul0,
            mul1,
            mul0.stride(0),
            mul0.stride(1),
            mul1.stride(0),
            mul1.stride(1),
            batch,
            input_row,
            total_jobs,
            loop,
            slice_l,
            is_mul1_multi_dim,
            block_size_b0,
            block_size_r,
        )

    return output_tensor

@fused_mul_sum.register_fake
def fused_mul_sum_fake(
    mul0: torch.Tensor,
    mul1: torch.Tensor,
    slice_l: Optional[int],
) -> torch.Tensor:
    batch = mul0.shape[0]
    output_tensor = torch.zeros(
        (batch, 1),
        device=mul0.device,
        dtype=mul0.dtype,
    )
    return output_tensor
