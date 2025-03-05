import torch
import torch_npu
import triton
import triton.language as tl


from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta


@triton.jit
def npu_triton_slice_cat_kernel(
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

npu_def.define("fused_slice_cat(Tensor input_tensor, Tensor indices_tensor, int n_rows, int elements, int input_stride, int block_size) -> (Tensor)")
#@torch.library.custom_op("torch_npu_triton::fused_slice_cat", mutates_args=())
@impl(npu_lib, "fused_slice_cat")
def fused_slice_cat(
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
    # npu_triton_slice_cat_kernel[(n_rows, num_blocks)](
    #     input_tensor,
    #     output_tensor,
    #     indices_tensor,
    #     input_stride,
    #     elements,
    #     BLOCK_SIZE=block_size,
    # )
    return output_tensor


#@fused_slice_cat.register_fake
@impl(npu_meta, "fused_slice_cat")
def fused_slice_cat_fake(
    input_tensor, indices_tensor, n_rows, elements, input_stride, block_size
):
    return torch.empty(n_rows, elements, device=input_tensor.device)
