import torch
import torch_npu
import triton
import triton.language as tl
from typing import List

# @triton.jit
# def npu_triton_slice_low_kernel(
#     input_ptr,
#     output_ptr,
#     start_indices_ptr,
#     slice_len,
#     input_row,
#     input_stride: tl.constexpr,
#     BLOCK_SIZE_R: tl.constexpr = 16,
#     BLOCK_SIZE_C: tl.constexpr = 128,
# ):
#     slice_idx = tl.program_id(0)
#     start_index = tl.load(start_indices_ptr + slice_idx)
#     offset_c = tl.arange(0, BLOCK_SIZE_C)
#     offset_r = tl.arange(0, BLOCK_SIZE_R)
#     mask_c = offset_c < slice_len
#     mask_r = offset_r < input_row
#     mask = mask_r[:, None] & mask_c[None, :]
#     value = tl.load(
#         input_ptr
#         + offset_r[:, None] * input_stride
#         + (offset_c[None, :] + start_index),
#         mask=mask,
#     )

#     tl.store(
#         output_ptr
#         + slice_idx * input_row * slice_len
#         + offset_r[:, None] * slice_len
#         + offset_c[None, :],
#         value,
#         mask=mask,
#     )


@triton.jit
def npu_triton_slice_low_kernel(
    input_ptr,
    output_ptr,
    start_indices_ptr,
    slice_len,
    input_row,
    input_stride: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr = 1024,
    BLOCK_SIZE_C: tl.constexpr = 128,
    BLOCK_SIZE_R_SUB: tl.constexpr = 512,
):
    slice_idx = tl.program_id(0)
    start_index = tl.load(start_indices_ptr + slice_idx)
    offset_c = tl.arange(0, BLOCK_SIZE_C)
    offset_r_sub = tl.arange(0, BLOCK_SIZE_R_SUB)
    mask_c = offset_c < slice_len
    r_sub_loops = (BLOCK_SIZE_R + BLOCK_SIZE_R_SUB - 1) // BLOCK_SIZE_R_SUB
    for r_sub_id in tl.range(0, r_sub_loops):
        offset_r = offset_r_sub + r_sub_id * BLOCK_SIZE_R_SUB
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

from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta

npu_def.define("fused_slice_low(Tensor src_tensor, Tensor start_indices, int slice_len, int n_rows, int input_stride) -> (Tensor)")
@impl(npu_lib, "fused_slice_low")
def fused_slice_low(
    src_tensor: torch.Tensor,
    start_indices: torch.Tensor,
    slice_len: int,
    n_rows: int,
    input_stride: int,
) -> torch.Tensor:
    block_size_r = ((n_rows + 16 - 1) // 16) * 16
    block_size_c = ((slice_len + 128 - 1) // 128) * 128
    UB_SIZE_IN_BYTES = 192 * 1024
    UB_SIZE_IN_NUMEL = UB_SIZE_IN_BYTES // src_tensor.element_size()
    block_size_r_sub = UB_SIZE_IN_NUMEL // block_size_c
    # To avoid UB overflow when the compiler enables DB as default,
    # we further divides by 2.
    block_size_r_sub = block_size_r_sub // 2
    output_tensors = torch.empty(
        (len(start_indices), src_tensor.shape[0], slice_len),
        device=src_tensor.device,
        dtype=src_tensor.dtype,
    )
    num_slices = len(start_indices)
    grid = (num_slices, 1, 1)

    if not (type(src_tensor) is torch._subclasses.fake_tensor.FakeTensor):
        npu_triton_slice_low_kernel[grid](
            src_tensor,
            output_tensors,
            start_indices,
            slice_len,
            n_rows,
            input_stride,
            block_size_r,
            block_size_c,
            block_size_r_sub,
        )

    return output_tensors


# @fused_slice_low.register_fake
@impl(npu_meta, "fused_slice_low")
def fused_slice_low_fake(
    src_tensor, start_indices, slice_len, n_rows, input_stride
):
    output_tensors = torch.empty(
        (len(start_indices), src_tensor.shape[0], slice_len),
        device=src_tensor.device,
        dtype=src_tensor.dtype,
    )
    return output_tensors
