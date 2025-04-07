import pytest
import math

import torch
import xpu_graph

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)
import triton
import triton.language as tl

@triton.jit
def single_mul_sum_cat(
    mul0,
    mul1,
    output,
    mul_stride0,
    mul_stride1,
    output_stride,
    input_row,
    slice_len,
    row_start_idx,
    output_start_offset,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 16,
    BLOCK_SIZE_S1: tl.constexpr = 128,
    BLOCK_SIZE_S2: tl.constexpr = 128,
):
    input_block_ptr0 = tl.make_block_ptr(
        base=mul0,
        shape=(input_row, slice_len),
        strides=(mul_stride0, 1),
        offsets=(row_start_idx, 0),
        block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
        order=(1, 0),
    )
    input_block_ptr1 = tl.make_block_ptr(
        base=mul1,
        shape=(input_row, slice_len),
        strides=(mul_stride1, 1),
        offsets=(row_start_idx, 0),
        block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
        order=(1, 0),
    )
    value0 = tl.load(input_block_ptr0, boundary_check=(0,), padding_option=0)
    value1 = tl.load(input_block_ptr1, boundary_check=(0,), padding_option=0)
    value0 = value0 * value1
    value0 = tl.reshape(value0, [BLOCK_SIZE_R, BLOCK_SIZE_S1, BLOCK_SIZE_S2])
    value0 = tl.sum(value0, axis=1)
    output_block_ptr = tl.make_block_ptr(
        base=output,
        shape=(input_row, BLOCK_SIZE_S2 * 2),
        strides=(output_stride, 1),
        offsets=(row_start_idx, output_start_offset),
        block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_S2),
        order=(1, 0),
    )
    tl.store(output_block_ptr, value0, boundary_check=(0,))

@triton.jit
def mlu_triton_mul_sum_cat_kernel(
    mul0,
    mul1,
    mul2,
    mul3,
    output,
    mul_stride0,
    mul_stride1,
    mul_stride2,
    mul_stride3,
    output_stride,
    total_jobs,
    input_row,
    slice_len,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 16,
    BLOCK_SIZE_S1: tl.constexpr = 128,
    BLOCK_SIZE_S2: tl.constexpr = 128,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    block_jobs = total_jobs // program_dim
    remainder = total_jobs % program_dim
    # by row(batch)
    if program_id < remainder:
        block_jobs_r = block_jobs + 1
        offset = program_id * (block_jobs + 1)
    else:
        block_jobs_r = block_jobs
        offset = remainder * (block_jobs + 1) + (program_id - remainder) * block_jobs

    loop = (block_jobs_r + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R
    for l in range(loop):
        row_start_idx = offset + l * BLOCK_SIZE_R
        single_mul_sum_cat(
            mul0,
            mul1,
            output,
            mul_stride0,
            mul_stride1,
            output_stride,
            input_row,
            slice_len,
            row_start_idx,
            0,
            BLOCK_SIZE_R,
            BLOCK_SIZE_C,
            BLOCK_SIZE_S1,
            BLOCK_SIZE_S2,
        )
        single_mul_sum_cat(
            mul2,
            mul3,
            output,
            mul_stride0,
            mul_stride1,
            output_stride,
            input_row,
            slice_len,
            row_start_idx,
            BLOCK_SIZE_S2,
            BLOCK_SIZE_R,
            BLOCK_SIZE_C,
            BLOCK_SIZE_S1,
            BLOCK_SIZE_S2,
        )


@torch.library.custom_op("torch_mlu_triton::fused_mul_sum_cat", mutates_args=())
def fused_mul_sum_cat_2inp(
    mul1: torch.Tensor,
    mul2: torch.Tensor,
    mul3: torch.Tensor,
    mul4: torch.Tensor,
) -> torch.Tensor:
    src_tensor = mul1
    s0, s1, s2 = mul1.shape
    block_size_r = s0
    block_size_c = s1 * s2
    size_of_dtype = 2
    if src_tensor.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = 384 * 1024
    if block_size_r * block_size_c * size_of_dtype * 4 > nram_limit:
        block_size_r = nram_limit // (size_of_dtype * block_size_c * 4)
    #block_size_r 3 128 128 16384
    print("block_size_r", block_size_r, s1, s2, s1 * s2)

    output_tensors = torch.zeros(
        (s0, s2 * 2),
        device=src_tensor.device,
        dtype=src_tensor.dtype,
    )
    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    grid = (processor_count, 1, 1)
    mlu_triton_mul_sum_cat_kernel[grid](
        mul1.view(s0, -1),
        mul2.view(s0, -1),
        mul3.view(s0, -1),
        mul4.view(s0, -1),
        output_tensors,
        mul1.stride(0),
        mul2.stride(0),
        mul3.stride(0),
        mul4.stride(0),
        output_tensors.stride(0),
        s0,
        s0,
        s1 * s2,  # slice_len,
        block_size_r,
        s1 * s2,
        s1,
        s2
    )

    return output_tensors


def fn0(x1, x2, x3, x4):
    a = x1 * x2
    sum_a = a.sum(dim=1)
    b = x3 * x4
    sum_b = b.sum(dim=1)
    out = torch.cat([sum_a, sum_b], dim=1)
    return out


def bmm_test(xpu_graph_backend, func):
    batch = 2048
    dtype = torch.half
    a = torch.randn(batch, 128, 128, dtype=dtype, device="mlu")
    b = torch.randn(batch, 128, 128, dtype=dtype, device="mlu")
    c = torch.randn(batch, 128, 128, dtype=dtype, device="mlu")
    d = torch.randn(batch, 128, 128, dtype=dtype, device="mlu")

    res1 = func(a, b, c, d)
    # compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    # res = compiled(a,b,c,d)
    res = fused_mul_sum_cat_2inp(a,b,c,d)
    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
        freeze=True,
        opt_level=OptLevel.level2,
        vendor_compiler_config=False,
        debug=True,
    )
    bmm_test(xpu_graph_backend, fn0)
