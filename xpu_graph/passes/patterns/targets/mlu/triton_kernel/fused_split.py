from typing import Optional, Tuple, List
import torch
import sys
sys.path.append("/data01/jyj/0327/modelcode/xpu_graph/xpu_graph/passes/patterns/targets/mlu/triton_kernel")
from fused_slice import mlu_triton_slice_low_kernel 

# Causal Conv1D Forward Function
@torch.library.custom_op(
    "torch_mlu_triton::fused_split_fwd",
    mutates_args=(),
    device_types="mlu",
)
def fused_split_fwd(
    x: torch.Tensor, 
    split_size: int,
    dim:int,
) -> torch.Tensor:
    split_num = x.shape[1] // split_size
    assert(split_num * split_size == x.shape[1])
    slices_index = list(range(0, x.shape[1], split_size))
    start_indices = torch.tensor(slices_index, device = x.device, dtype = torch.int32)

    n_rows = x.shape[0]
    input_stride = x.stride(0)
    slice_len = split_size
    block_size_r = n_rows
    block_size_c = slice_len
    size_of_dtype = 2
    if x.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = 384 * 1024
    if block_size_r * block_size_c * size_of_dtype > nram_limit:
        block_size_r = nram_limit // size_of_dtype // block_size_c

    num_slices = len(start_indices)
    output_tensors = torch.empty(
        (num_slices * x.shape[0], slice_len),
        device=x.device,
        dtype=x.dtype,
        requires_grad=x.requires_grad,
    )
    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    grid = (processor_count, 1, 1)
    mlu_triton_slice_low_kernel[grid](
        x,
        output_tensors,
        start_indices,
        num_slices,
        slice_len,
        n_rows,
        input_stride,
        block_size_r,
        block_size_c,
    )

    return output_tensors.view(num_slices, x.shape[0], slice_len).unbind(0)



# Register a fake forward pass for tracing
@fused_split_fwd.register_fake
def fused_split_fwd(
    x: torch.Tensor, 
    split_size: int,
    dim:int,
) -> Tuple[torch.Tensor]:
    split_num = x.shape[1] // split_size
    output_tensors = torch.empty(
        (split_num, x.shape[0], split_size),
        device=x.device,
        dtype=x.dtype,
        requires_grad=x.requires_grad
    )
    outputs = output_tensors.unbind(0) 
    #print("Forward outputs require grad:", [t.requires_grad for t in outputs])
    return outputs 

# Causal Conv1D Backward Function
@torch.library.custom_op(
    "torch_mlu_triton::fused_split_bwd", 
    mutates_args=(),
    device_types="mlu",
)
def fused_split_bwd(
    #grad: Tuple[torch.Tensor],
    grads: List[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    assert len(grads) > 0, "Empty gradients in backward"
    return torch.cat(grads, dim=dim)

# Register a fake backward pass for tracing
@fused_split_bwd.register_fake
def _fused_split_bwd_fake(
    grad: Tuple[torch.Tensor],
    dim: int,
):
    return torch.cat(grads, dim=dim), None, None

# Setup context for autograd
def fused_splitsetup_context(ctx, inputs, output):
    x, split_size, dim = inputs
    ctx.save_for_backward(dim)

# Bridge for backward pass in autograd
def fused_split_bwd_bridge(ctx, dout):
    dim = ctx.saved_tensors
    return fused_split_bwd(dout, dim)

# Register custom autograd function
torch.library.register_autograd(
    "torch_mlu_triton::fused_split_fwd",
    fused_split_bwd_bridge,
    setup_context=fused_splitsetup_context,
)


class FusedSplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, split_size: int, dim: int):
        ctx.split_size = split_size
        ctx.dim = dim
        return fused_split_fwd(x, split_size, dim)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        dim = ctx.dim
        grad_x = fused_split_bwd(tuple(grads), dim)
        return grad_x, None, None

# 包装函数
def fused_split(x: torch.Tensor, split_size: int, dim: int = -1):
    return FusedSplitFunction.apply(x, split_size, dim)

# Define a higher-level function to invoke the custom op
def fused_splitfn(x,split_size,dim):
    output = fused_split(x, split_size, dim)
    return output[0] + output[1] + output[2] + output[3] 


# Test the implementation
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 32, device="mlu:0", requires_grad=True)

    # Test the forward and backward pass
    print("Custom Implementation")
    out = fused_splitfn(x, 8, 1)
    print("out.requires_grad:", out.requires_grad)  # 应为 True
    print("out.grad_fn:", out.grad_fn)             # 应显示自定义反向函数

    print(out.sum())
    out.sum().backward()
    print("1", x.grad)

    print(out.min(), out.max(), out.mean(), out.std())
    print(x.grad.min(), x.grad.max(), x.grad.mean(), x.grad.std())

    # Try compiling the function using torch.compile
    x.grad.zero_()
    compiled_conv1d = torch.compile(fused_splitfn)

    # Run the compiled function
    print("Compiled Implementation")
    out = compiled_conv1d(x.clone(), 8, 1)
    print("out.requires_grad:", out.requires_grad)  # 应为 True
    print("out.grad_fn:", out.grad_fn)             # 应显示自定义反向函数
    out.sum().backward()

    print(out.min(), out.max(), out.mean(), out.std())
    #print(x.grad.min(), x.grad.max(), x.grad.mean(), x.grad.std())

