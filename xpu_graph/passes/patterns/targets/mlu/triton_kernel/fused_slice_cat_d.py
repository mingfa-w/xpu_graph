import torch
import torch_mlu
import triton
import triton.language as tl


@triton.jit
def mlu_triton_slice_cat_d_kernel(
    input_ptrs,
    output_ptr,
    indices_ptr,
    slice_lens,
    output_offsets,
    total_jobs,
    BLOCK_SIZE_R: tl.constexpr = 32,
    BLOCK_SIZE_C: tl.constexpr = 32,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    block_jobs = total_jobs // program_dim
    remain = total_jobs % program_dim
    block_jobs_r = block_jobs
    offset = remain * (block_jobs + 1) + (program_id - remain) * block_jobs
    if program_id < remain:
        block_jobs_r += 1
        offset = program_id * (block_jobs + 1)

    tl.device_print("X",offset)

@torch.library.custom_op("torch_mlu_triton::fused_slice_cat_d", mutates_args=())
def fused_slice_cat_d(
    input_tensor_ptrs: torch.Tensor,
    start_indices: torch.Tensor,
    slice_lens: torch.Tensor,
    output_offsets: torch.Tensor,
    batch_size: int,
    output_len: int,
    max_slice_len: int,
) -> torch.Tensor:
    print("?")
    num_blocks = (elements + block_size - 1) // block_size
    output_tensor = torch.empty(batch_size, output_len, device=input_tensor.device)

    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    grid = (processor_count, 1, 1)

    mlu_triton_slice_cat_d_kernel[grid](
        input_tensor_ptrs,
        output_tensor,
        start_indices,
        slice_lens,
        output_offsets,
        len(slice_lens),
        batch_size,
        max_slice_len
    )
    exit(-1)
    return output_tensor


@fused_slice_cat_d.register_fake
def fused_slice_cat_d_fake(
    input_tensor_ptrs: torch.Tensor,
    start_indices: torch.Tensor,
    slice_lens: torch.Tensor,
    output_offsets: torch.Tensor,
    batch_size: int,
    output_len: int,
    max_slice_len: int,
) -> torch.Tensor:
    return torch.empty(batch_size, output_len, device=input_tensor.device)
