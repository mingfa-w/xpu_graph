import torch
import torch_mlu
import triton
import triton.language as tl
from typing import List


@triton.jit
def mlu_triton_slice_low_kernel(
    input_ptr,
    output_ptr,
    start_indices_ptr,
    slice_len,
    input_row,
    input_stride,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 128,
):
    slice_idx = tl.program_id(0)
    start_index = tl.load(start_indices_ptr + slice_idx)
    offset_c = tl.arange(0, BLOCK_SIZE_C)
    offset_r = tl.arange(0, BLOCK_SIZE_R)
    mask_c = offset_c < slice_len
    mask_r = offset_r < input_row
    mask = mask_r[:, None] & mask_c[None, :]
    value = tl.load(
        input_ptr
        + offset_r[:, None] * input_stride
        + (offset_c[None, :] + start_index),
        mask=mask,
    )

    tl.store(
        output_ptr
        + slice_idx * input_row * slice_len
        + offset_r[:, None] * slice_len
        + offset_c[None, :],
        value,
        mask=mask,
    )


@torch.library.custom_op("torch_mlu_triton::fused_slice_low", mutates_args=())
def fused_slice_low(
    src_tensor: torch.Tensor,
    start_indices: torch.Tensor,
    slice_len: int,
    n_rows: int,
    input_stride: int,
) -> torch.Tensor:
    block_size_r = ((n_rows + 16 - 1) // 16) * 16
    block_size_c = ((slice_len + 128 - 1) // 128) * 128
    output_tensors = torch.empty(
        (len(start_indices), src_tensor.shape[0], slice_len),
        device=src_tensor.device,
        dtype=src_tensor.dtype,
    )
    num_slices = len(start_indices)
    grid = (num_slices, 1, 1)
    mlu_triton_slice_low_kernel[grid](
        src_tensor,
        output_tensors,
        start_indices,
        slice_len,
        n_rows,
        input_stride,
        block_size_r,
        block_size_c,
    )

    return output_tensors


@fused_slice_low.register_fake
def fused_slice_low_fake(src_tensor, start_indices, slice_len, n_rows, input_stride):
    output_tensors = torch.empty(
        (len(start_indices), src_tensor.shape[0], slice_len),
        device=src_tensor.device,
        dtype=src_tensor.dtype,
    )
    return output_tensors
