import math
import torch
import torch_npu
import triton
import triton.language as tl
from typing import List

dtype_dict1 = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
}


# @triton.jit
# def npu_triton_slice_sum_cat_kernel(
#     output_ptr,
#     input_ptr,
#     indices_ptr,
#     input_stride0,
#     input_stride1,
#     output_stride0,
#     row,
#     col,
#     output_num,
#     BLOCK_SIZE_T: tl.constexpr = 32,
#     BLOCK_SIZE_I: tl.constexpr = 32,
#     BLOCK_SIZE_R: tl.constexpr = 32,
#     BLOCK_SIZE_C: tl.constexpr = 32,
# ):
#     program_id = tl.program_id(0)
#     program_dim = tl.num_programs(axis=0)
#     indices_offset = indices_ptr
#     input_offset = input_ptr + program_id * input_stride0
#     output_offset = output_ptr + program_id * output_stride0

#     offset_param = tl.arange(0, BLOCK_SIZE_I)
#     offset_row = tl.arange(0, BLOCK_SIZE_R)
#     offset_col = tl.arange(0, BLOCK_SIZE_C)

#     mask_row = offset_row < row
#     mask_col = offset_col < col
#     mask = mask_row[:, None] & mask_col[None, :]
#     mask_indices = offset_param < output_num * 2
#     indices = tl.load(indices_offset + offset_param, mask=mask_indices)
#     input_data = tl.load(
#         input_offset + offset_row[:, None] * input_stride1 + offset_col[None, :],
#         mask=mask,
#     )
#     for i in range(output_num):
#         indices_start = 0#indices[i * 2]
#         indices_end = 8#indices[i * 2 + 1]
#         slice_mask = (offset_row < indices_end) & (offset_row > (indices_start - 1))
#         combined_mask = slice_mask[:, None] & mask_col[None, :]
#         slice_value = tl.where(combined_mask, input_data, 0)
#         sum_value = tl.sum(slice_value, axis=0)
#         tl.store(output_offset + i * col + offset_col, sum_value, mask=mask_col)


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
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    program_id = tl.program_id(0)
    # program_id = 85
    indices_offset = indices_ptr
    input_offset = input_ptr + program_id * input_stride0
    output_offset = output_ptr + program_id * output_stride0

    index_row = tl.arange(0, BLOCK_SIZE_R)
    # mask_row = index_row < row
    for coffs in range(0,col,BLOCK_SIZE_C):
        index_col = tl.arange(0, BLOCK_SIZE_C) + coffs
        mask_col = index_col < col
        # # origin code :
        # mask = mask_row[:, None] & mask_col[None, :]
        # input_data = tl.load(
        #     input_offset + index_row[:, None] * input_stride1 + index_col[None, :],
        #     mask=mask,
        # )
        for i in range(output_num):
            indices_start = tl.load(indices_offset + i*2)
            indices_end = tl.load(indices_offset + i*2 + 1)

            true_index_row = index_row[:,None] + indices_start

            slice_value = tl.load(
                input_offset + true_index_row * input_stride1 + index_col[None,:],
                mask = (true_index_row < indices_end) & mask_col,
                other = 0
            )
            # # origin code : 
            # slice_mask = (index_row < indices_end) & (index_row > (indices_start - 1))
            # combined_mask = slice_mask[:, None] & mask_col[None, :]
            # slice_value = tl.where(combined_mask, input_data, 0)
            
            sum_value = tl.sum(slice_value, axis=0)
            # pdb.set_trace()
            tl.store(output_offset + i * col + index_col, sum_value, mask=mask_col)

#from torch.library import Library, impl
# from ..fused_sum_cat import npu_def, npu_lib, npu_meta
# npu_def = Library("torch_npu_triton", "DEF")
# npu_lib = Library("npu_graph", "IMPL", "PrivateUse1")
# npu_meta =Library("npu_graph", "IMPL", "Meta")
from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta
npu_def.define("fuse_slice_sum_cat(Tensor input_tensor, Tensor slice_tensor, int processor_count, int output_num) -> (Tensor)")

# @torch.library.custom_op(
#     "torch_npu_triton::fuse_slice_sum_cat", mutates_args=(), device_types="npu"
# )
@impl(npu_lib, "fuse_slice_sum_cat")
def fuse_slice_sum_cat(
    input_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    processor_count: int,
    output_num: int,
) -> torch.Tensor:
    # output_tensor = torch.empty(
    #     [input_tensor.shape[0], output_num * input_tensor.shape[2]],
    #     dtype=input_tensor.dtype,
    #     device=input_tensor.device,
    # )
    # input_stride0 = input_tensor.stride(0)
    # input_stride1 = input_tensor.stride(1)
    # batch = input_tensor.shape[0]
    # row = input_tensor.shape[1]
    # col = input_tensor.shape[2]
    # total_jobs = batch
    # mini_block = 4
    # block_size_t = 1
    # block_size_i = ((output_num * 2 + mini_block - 1) // mini_block) * mini_block
    # block_size_r = ((row + mini_block - 1) // mini_block) * mini_block
    # block_size_c = ((col + mini_block - 1) // mini_block) * mini_block
    # grid = (total_jobs, 1, 1)
    # # npu_triton_slice_sum_cat_kernel[grid](
    # #     output_tensor,
    # #     input_tensor,
    # #     slice_tensor,
    # #     input_stride0,
    # #     input_stride1,
    # #     output_tensor.stride(0),
    # #     row,
    # #     col,
    # #     output_num,
    # #     block_size_t,
    # #     block_size_i,
    # #     block_size_r,
    # #     block_size_c,
    # #     num_stages=3,
    # # )
    # return output_tensor
    
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
    grid = (total_jobs, 1, 1)

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
        block_size_r,
        block_size_c,
    )
    return output_tensor


# @fuse_slice_sum_cat.register_fake
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
