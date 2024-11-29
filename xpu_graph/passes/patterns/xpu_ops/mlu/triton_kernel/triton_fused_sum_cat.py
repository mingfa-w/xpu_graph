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
def triton_fused_sumcat_2(
    input_ptrs,
    output_tensor,
    dim_0,
    dim_1_tensor,
    dim_2,
    num_tensors,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    block_id = tl.program_id(0)
    if DTYPE == 3:
        dtype = tl.float32
    elif DTYPE == 2:
        dtype = tl.float16
    elif DTYPE == 1:
        dtype = tl.bfloat16
    input_ptr = tl.load(input_ptrs + block_id).to(tl.pointer_type(dtype))
    dim_1 = tl.load(dim_1_tensor + block_id)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < dim_2 * dim_1
    for i_0 in range(dim_0):
        tensor_data = tl.load(
            input_ptr + i_0 * dim_2 * dim_1 + offset, mask=mask, other=0.0
        )
        result = tl.full([BLOCK_SIZE], 0.0, dtype=dtype)
        for i in range(dim_1):
            for j in range(dim_2):
                result[j] = result[j] + tensor_data[i * dim_2 + j]
        tl.store(
            output_tensor + block_id * dim_2 + i_0 * num_tensors * dim_2 + offset,
            result,
            mask=(offset < dim_2),
        )


@triton.jit
def triton_fused_sumcat_1(
    input_ptrs,
    output_tensor,
    dim_0,
    output_stride,
    input_lens,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    block_id = tl.program_id(0)
    if DTYPE == 3:
        dtype = tl.float32
    elif DTYPE == 2:
        dtype = tl.float16
    elif DTYPE == 1:
        dtype = tl.bfloat16
    input_ptr = tl.load(input_ptrs + block_id).to(tl.pointer_type(dtype))
    input_len = tl.load(input_lens + block_id)
    offset = tl.arange(0, BLOCK_SIZE)

    result = tl.full([BLOCK_SIZE], 0.0, dtype=dtype)
    for i in range(dim_0):
        tensor_data = tl.load(
            input_ptr + i * input_len + offset,
            mask=(offset < input_len),
        )
        tensor_sum = tl.sum(tensor_data, axis=0)
        result[i] = tensor_sum

    tl.store(
        output_tensor + block_id * output_stride + offset, result, mask=(offset < dim_0)
    )


@torch.library.custom_op(
    "aten::torch_triton_fused_sumcat_replacement1", mutates_args=()
)
def torch_triton_fused_sumcat_replacement1(
    inputs: List[torch.Tensor],
    dim_0: int,
    output_stride: int,
    input_lens: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    device = inputs[0].device
    dtype = inputs[0].dtype
    input_tensor = torch.tensor([t.data_ptr() for t in inputs], device=device)
    output_tensor = torch.empty(
        (input_lens.shape[0], dim_0),
        device=device,
        dtype=dtype,
    )
    grid = (output_tensor.shape[0], 1, 1)
    triton_fused_sumcat_1[grid](
        input_tensor,
        output_tensor,
        dim_0,
        output_stride,
        input_lens,
        BLOCK_SIZE=block_size,
        DTYPE=dtype_dict1[dtype],
    )
    return output_tensor


@torch_triton_fused_sumcat_replacement1.register_fake
def torch_triton_fused_sumcat_replacement1_fake(
    inputs: List[torch.Tensor],
    dim_0: int,
    output_stride: int,
    input_lens: torch.Tensor,
    block_size=1024,
):
    device = inputs[0].device
    dtype = inputs[0].dtype
    return torch.empty(input_lens.shape[0], dim_0, device=device, dtype=dtype)


@torch.library.custom_op(
    "aten::torch_triton_fused_sumcat_replacement2", mutates_args=()
)
def torch_triton_fused_sumcat_replacement2(
    inputs: List[torch.Tensor],
    dim_0: int,
    dim_1_tensor: torch.Tensor,
    dim_2: int,
    num_tensors: int,
) -> torch.Tensor:
    device = inputs[0].device
    dtype = inputs[0].dtype
    output_tensor = torch.empty(dim_0, dim_2 * num_tensors, device=device, dtype=dtype)
    ptrs = [t.data_ptr() for t in inputs]
    tensor_ptrs = torch.tensor(ptrs, device=device)
    grid = (num_tensors, 1, 1)
    triton_fused_sumcat_2[grid](
        tensor_ptrs,
        output_tensor,
        dim_0,
        dim_1_tensor,
        dim_2,
        num_tensors,
        128 * 128,
        DTYPE=dtype_dict1[dtype],
    )
    return output_tensor


@torch_triton_fused_sumcat_replacement2.register_fake
def torch_triton_fused_sumcat_replacement2_fake(
    inputs: List[torch.Tensor],
    dim_0: int,
    dim_1: torch.Tensor,
    dim_2: int,
    num_tensors: int,
):
    device = inputs[0].device
    dtype = inputs[0].dtype
    output_tensor = torch.empty(dim_0, dim_2 * num_tensors, device=device, dtype=dtype)
    return output_tensor
