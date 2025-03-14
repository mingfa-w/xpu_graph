import torch
import torch_npu
import triton
import triton.language as tl


from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta


@triton.jit
def do_special_slice_cat(
    in_ptr,
    out_ptr,
    in_offset_ptr, out_offset_ptr,
    cnt : tl.constexpr,
    size : tl.constexpr,
):
    idx = tl.arange(0,size)
    for i in range(cnt):
        in_offset=tl.load(in_offset_ptr+i)
        out_offset=tl.load(out_offset_ptr+i)
        val=tl.load(in_ptr+in_offset+idx)
        tl.store(out_ptr+out_offset+idx,val)

# {64: 436, 16: 1116, 8: 1294, 4: 448, 32: 1176, 1: 224}
# indices continous size=1,4,8,16,32,64
# point 1 : should cnt_* be a constant?
# point 2 : should there be an 'if' before calling 'do_special_slice_cat' 
# point 3 : gather ?
@triton.jit
def npu_triton_slice_cat_kernel_4_qianchuan(
    data_ptr,out_ptr,
    in_1_offs_ptr,out_1_offs_ptr,
    in_4_offs_ptr,out_4_offs_ptr,
    in_8_offs_ptr,out_8_offs_ptr,
    in_16_offs_ptr,out_16_offs_ptr,
    in_32_offs_ptr,out_32_offs_ptr,
    in_64_offs_ptr,out_64_offs_ptr,
    stride : tl.constexpr,
    output_xnumel : tl.constexpr,
    cnt_1 : tl.constexpr,
    cnt_4 : tl.constexpr,
    cnt_8 : tl.constexpr,
    cnt_16 : tl.constexpr,
    cnt_32 : tl.constexpr,
    cnt_64 : tl.constexpr,
    sz_1 : tl.constexpr = 1,
    sz_4 : tl.constexpr = 4,
    sz_8 : tl.constexpr = 8,
    sz_16 : tl.constexpr = 16,
    sz_32 : tl.constexpr = 32,
    sz_64 : tl.constexpr = 64,
):
    row_id = tl.program_id(0)
    cur_data_ptr = data_ptr + row_id * stride
    cur_out_ptr = out_ptr + row_id * output_xnumel

    if cnt_1 > 0:
        do_special_slice_cat(cur_data_ptr,cur_out_ptr,in_1_offs_ptr,out_1_offs_ptr,cnt_1,sz_1)
    
    if cnt_4 > 0:
        do_special_slice_cat(cur_data_ptr,cur_out_ptr,in_4_offs_ptr,out_4_offs_ptr,cnt_4,sz_4)
    
    if cnt_8 > 0:
        do_special_slice_cat(cur_data_ptr,cur_out_ptr,in_8_offs_ptr,out_8_offs_ptr,cnt_8,sz_8)

    if cnt_16 > 0:
        do_special_slice_cat(cur_data_ptr,cur_out_ptr,in_16_offs_ptr,out_16_offs_ptr,cnt_16,sz_16)

    if cnt_32 > 0:
        do_special_slice_cat(cur_data_ptr,cur_out_ptr,in_32_offs_ptr,out_32_offs_ptr,cnt_32,sz_32)

    if cnt_64 > 0:
        do_special_slice_cat(cur_data_ptr,cur_out_ptr,in_64_offs_ptr,out_64_offs_ptr,cnt_64,sz_64)


npu_def.define((
    "fused_slice_cat("
    "Tensor input_tensor, int n_rows,"
    "Tensor in_offs_ptr0, Tensor out_offs_ptr0,"
    "Tensor in_offs_ptr1, Tensor out_offs_ptr1,"
    "Tensor in_offs_ptr2, Tensor out_offs_ptr2,"
    "Tensor in_offs_ptr3, Tensor out_offs_ptr3,"
    "Tensor in_offs_ptr4, Tensor out_offs_ptr4,"
    "Tensor in_offs_ptr5, Tensor out_offs_ptr5,"
    "int stride,"
    "int output_xnumel,"
    "int cnt_1,"
    "int cnt_4,"
    "int cnt_8,"
    "int cnt_16,"
    "int cnt_32,"
    "int cnt_64,"
    "int sz_1,"
    "int sz_4,"
    "int sz_8,"
    "int sz_16,"
    "int sz_32,"
    "int sz_64"
    ")-> (Tensor)"
))
@impl(npu_lib, "fused_slice_cat")
def fused_slice_cat(
    input_tensor,n_rows,
    in_offs_ptr0,out_offs_ptr0,
    in_offs_ptr1,out_offs_ptr1,
    in_offs_ptr2,out_offs_ptr2,
    in_offs_ptr3,out_offs_ptr3,
    in_offs_ptr4,out_offs_ptr4,
    in_offs_ptr5,out_offs_ptr5,
    stride,
    output_xnumel,
    cnt_1,
    cnt_4,
    cnt_8,
    cnt_16,
    cnt_32,
    cnt_64,
    sz_1,
    sz_4,
    sz_8,
    sz_16,
    sz_32,
    sz_64,
) -> torch.Tensor:

    output_tensor = torch.empty(n_rows, output_xnumel, device=input_tensor.device, dtype=input_tensor.dtype)

    npu_triton_slice_cat_kernel_4_qianchuan[n_rows,1,1](
        input_tensor,output_tensor,
        in_offs_ptr0,out_offs_ptr0,
        in_offs_ptr1,out_offs_ptr1,
        in_offs_ptr2,out_offs_ptr2,
        in_offs_ptr3,out_offs_ptr3,
        in_offs_ptr4,out_offs_ptr4,
        in_offs_ptr5,out_offs_ptr5,
        stride=stride,
        output_xnumel=output_xnumel,
        cnt_1=cnt_1,
        cnt_4=cnt_4,
        cnt_8=cnt_8,
        cnt_16=cnt_16,
        cnt_32=cnt_32,
        cnt_64=cnt_64,
        sz_1=sz_1,
        sz_4=sz_4,
        sz_8=sz_8,
        sz_16=sz_16,
        sz_32=sz_32,
        sz_64=sz_64,
    )

    return output_tensor

#@fused_slice_cat.register_fake
@impl(npu_meta, "fused_slice_cat")
def fused_slice_cat_fake(
    input_tensor,n_rows,
    in_offs_ptr0,out_offs_ptr0,
    in_offs_ptr1,out_offs_ptr1,
    in_offs_ptr2,out_offs_ptr2,
    in_offs_ptr3,out_offs_ptr3,
    in_offs_ptr4,out_offs_ptr4,
    in_offs_ptr5,out_offs_ptr5,
    stride,
    output_xnumel,
    cnt_1,
    cnt_4,
    cnt_8,
    cnt_16,
    cnt_32,
    cnt_64,
    sz_1,
    sz_4,
    sz_8,
    sz_16,
    sz_32,
    sz_64,
):
    return torch.empty(n_rows, output_xnumel, device=input_tensor.device, dtype=input_tensor.dtype)
