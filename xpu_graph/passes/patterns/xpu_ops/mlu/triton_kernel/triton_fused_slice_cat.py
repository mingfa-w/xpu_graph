import torch
import torch_mlu
import triton
import triton.language as tl


@triton.jit
def mlu_triton_slice_cat_kernel(
    data_ptr, output_ptr, indices_ptr, stride, n_elements, BLOCK_SIZE: tl.constexpr
):
    block_id = tl.program_id(1)
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    indices = tl.load(indices_ptr + offset, mask=mask, other=0)

    row_id = tl.program_id(0)

    data_offset = row_id * stride + indices
    data = tl.load(data_ptr + data_offset, mask=mask, other=0)

    output_offset = row_id * n_elements + offset
    tl.store(output_ptr + output_offset, data, mask=mask)


@torch.library.custom_op("aten::mlu_triton_fused_slice_cat", mutates_args=())
def mlu_triton_fused_slice_cat(
    input_tensor: torch.Tensor,
    indices_tensor: torch.Tensor,
    n_rows: int,
    elements: int,
    input_stride: int,
    block_size: int,
) -> torch.Tensor:
    num_blocks = (elements + block_size - 1) // block_size
    output_tensor = torch.empty(
        (n_rows, elements), dtype=input_tensor.dtype, device=input_tensor.device
    )
    mlu_triton_slice_cat_kernel[(n_rows, num_blocks)](
        input_tensor,
        output_tensor,
        indices_tensor,
        input_stride,
        elements,
        BLOCK_SIZE=block_size,
    )
    return output_tensor


@mlu_triton_fused_slice_cat.register_fake
def mlu_triton_fused_slice_cat_fake(
    input_tensor, indices_tensor, n_rows, elements, input_stride, block_size
):
    return torch.empty(n_rows, elements, device=input_tensor.device)
