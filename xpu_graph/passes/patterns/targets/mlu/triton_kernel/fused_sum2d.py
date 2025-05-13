import torch
import torch_mlu
import triton
import triton.language as tl
from typing import List, Tuple
from .get_mlu_devinfo import get_device_properties


@triton.jit
def mlu_triton_sum_2d_dim_1_kernel(
    input_ptr,
    output_ptr,
    s0,
    s1,
    s2,
    BLOCK_SIZE_0: tl.constexpr = 16,
    BLOCK_SIZE_1: tl.constexpr = 16,
    BLOCK_SIZE_2: tl.constexpr = 16,
):
    loop = (s0 + BLOCK_SIZE_0 - 1) // BLOCK_SIZE_0
    for l in range(loop):
        offset_0 = l * BLOCK_SIZE_0
        input_ptr_ = tl.make_block_ptr(
            base=input_ptr,
            shape=(s0, s1, s2),
            strides=(s1 * s2, s2, 1),
            offsets=(offset_0, 0, 0),
            block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2),
            order=(2, 1, 0),
        )
        data = tl.load(input_ptr_, boundary_check=(2, 1, 0))
        data = tl.sum(data, axis=1)

        output_ptr_ = tl.make_block_ptr(
            base=output_ptr,
            shape=(s0, s2),
            strides=(s2, 1),
            offsets=(offset_0, 0),
            block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_2),
            order=(1, 0),
        )
        tl.store(output_ptr_, data, boundary_check=(1, 0))


@triton.jit
def mlu_triton_sum_2d_dim_2_kernel(
    input_ptr,
    output_ptr,
    s0,
    s1,
    s2,
    BLOCK_SIZE_0: tl.constexpr = 16,
    BLOCK_SIZE_1: tl.constexpr = 16,
    BLOCK_SIZE_2: tl.constexpr = 16,
):
    loop = (s0 + BLOCK_SIZE_0 - 1) // BLOCK_SIZE_0
    for l in range(loop):
        offset_0 = l * BLOCK_SIZE_0
        input_ptr_ = tl.make_block_ptr(
            base=input_ptr,
            shape=(s0, s1, s2),
            strides=(s1 * s2, s2, 1),
            offsets=(offset_0, 0, 0),
            block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2),
            order=(2, 1, 0),
        )
        data = tl.load(input_ptr_, boundary_check=(2, 1, 0))
        data = tl.sum(data, axis=2)

        output_ptr_ = tl.make_block_ptr(
            base=output_ptr,
            shape=(s0, s1),
            strides=(s1, 1),
            offsets=(offset_0, 0),
            block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_1),
            order=(1, 0),
        )
        tl.store(output_ptr_, data, boundary_check=(1, 0))


@triton.jit
def mlu_triton_sum_2d_kernel(
    input_ptrs,
    output_ptrs,
    tuple_s0,
    tuple_s1,
    tuple_s2,
    dim: tl.constexpr,
    TOTAL_JOBS: tl.constexpr = 16,
    BLOCK_SIZE_0: tl.constexpr = 16,
    BLOCK_SIZE_1: tl.constexpr = 16,
    BLOCK_SIZE_2: tl.constexpr = 16,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    for sum_idx in tl.static_range(TOTAL_JOBS):
        if sum_idx % program_dim == program_id:
            s0 = tuple_s0[sum_idx]
            s1 = tuple_s1[sum_idx]
            s2 = tuple_s2[sum_idx]
            input_ptr = input_ptrs[sum_idx]
            output_ptr = output_ptrs[sum_idx]
            if dim == 1:
                mlu_triton_sum_2d_dim_1_kernel(
                    input_ptr,
                    output_ptr,
                    s0,
                    s1,
                    s2,
                    BLOCK_SIZE_0,
                    BLOCK_SIZE_1,
                    BLOCK_SIZE_2,
                )
            else:
                mlu_triton_sum_2d_dim_2_kernel(
                    input_ptr,
                    output_ptr,
                    s0,
                    s1,
                    s2,
                    BLOCK_SIZE_0,
                    BLOCK_SIZE_1,
                    BLOCK_SIZE_2,
                )


@torch.library.custom_op("torch_mlu_triton::fused_sum_2d", mutates_args=())
def fused_sum_2d(
    inputs: List[torch.Tensor],
    dim: int,
) -> List[torch.Tensor]:
    props = get_device_properties()
    outputs = []
    max_s0 = max_s1 = max_s2 = 0
    for input in inputs:
        s0, s1, s2 = input.shape
        max_s0 = s0 if s0 > max_s0 else max_s0
        max_s1 = s1 if s1 > max_s1 else max_s1
        max_s2 = s2 if s2 > max_s2 else max_s2
        shape = ()
        if dim == 0:
            shape = (s1, s2)
        elif dim == 1:
            shape = (s0, s2)
        else:
            shape = (s0, s1)
        output = torch.empty(
            shape,
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )
        outputs.append(output)

    max_s1 = (max_s1 + 16 - 1) // 16 * 16
    max_s2 = (max_s2 + 16 - 1) // 16 * 16
    max_s0 = min(props.max_nram_size // (max_s1 * max_s2 * 8), max_s0)
    max_s0 = (max_s0 + 16 - 1) // 16 * 16

    # input must be contiguous
    mlu_triton_sum_2d_kernel[(props.total_cores,)](
        tuple([i.contiguous() for i in inputs]),
        tuple(outputs),
        tuple([i.shape[0] for i in inputs]),
        tuple([i.shape[1] for i in inputs]),
        tuple([i.shape[2] for i in inputs]),
        dim,
        len(inputs),
        int(max_s0),
        int(max_s1),
        int(max_s2),
    )
    return outputs


@fused_sum_2d.register_fake
def fused_sum_2d_fake(
    inputs: List[torch.Tensor],
    dim: int,
) -> List[torch.Tensor]:
    outputs = []
    for input in inputs:
        s0, s1, s2 = input.shape
        shape = ()
        if dim == 0:
            shape = (s1, s2)
        elif dim == 1:
            shape = (s0, s2)
        else:
            shape = (s0, s1)
        output = torch.empty(
            shape,
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )
        outputs.append(output)
    return outputs
'''
inputs = [
    torch.randn((1, 4, 1), dtype=torch.float32, device="mlu:0")
]
print(inputs)
output = fused_sum_2d(inputs, dim=1)
output1 = [torch.sum(input, dim=[1]) for input in inputs]
print(output1[0][0])
print(output[0][0])
exit()
'''
