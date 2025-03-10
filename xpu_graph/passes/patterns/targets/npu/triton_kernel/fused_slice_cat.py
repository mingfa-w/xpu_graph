import torch
import torch_npu
import triton
import triton.language as tl


from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta


# @triton.jit
# def npu_triton_slice_cat_kernel(
#     data_ptr, output_ptr, indices_ptr, stride, n_elements, BLOCK_SIZE: tl.constexpr
# ):
#     block_id = tl.program_id(1)
#     offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offset < n_elements

#     indices = tl.load(indices_ptr + offset, mask=mask, other=0)

#     row_id = tl.program_id(0)

#     data_offset = row_id * stride + indices
#     data = tl.load(data_ptr + data_offset, mask=mask, other=0)

#     output_offset = row_id * n_elements + offset
#     tl.store(output_ptr + output_offset, data, mask=mask)



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





# npu_def.define("fused_slice_cat(Tensor input_tensor, Tensor indices_tensor, int n_rows, int elements, int input_stride, int block_size) -> (Tensor)")
#@torch.library.custom_op("torch_npu_triton::fused_slice_cat", mutates_args=())
# @impl(npu_lib, "fused_slice_cat")
# def fused_slice_cat(
#     input_tensor: torch.Tensor,
#     indices_tensor: torch.Tensor,
#     n_rows: int,
#     elements: int,
#     input_stride: int,
#     block_size: int,
# ) -> torch.Tensor:
#     num_blocks = (elements + block_size - 1) // block_size
#     output_tensor = torch.empty(
#         (n_rows, elements), dtype=input_tensor.dtype, device=input_tensor.device
#     )
#     #import pdb;pdb.set_trace()
#     if not (type(input_tensor) is torch._subclasses.fake_tensor.FakeTensor):
#         npu_triton_slice_cat_kernel[(n_rows, num_blocks)](
#             input_tensor,
#             output_tensor,
#             indices_tensor,
#             input_stride,
#             elements,
#             BLOCK_SIZE=block_size,
#         )
#     return output_tensor
npu_def.define("fused_slice_cat(Tensor input_tensor, int[] int_list, int elements, int n_rows, int input_stride) -> (Tensor)")
@impl(npu_lib, "fused_slice_cat")
def fused_slice_cat(
    input_tensor,
    slices,
    elements,
    n_rows,
    input_stride,
) -> torch.Tensor:
    #import pdb;pdb.set_trace()
    count = {}
    in_offs_ptrs={}
    out_offs_ptrs={}
    position=0
    for i in range(0,len(slices),2):
        start=slices[i]
        end=slices[i+1]
        length = end - start
        if length in count:
            count[length]+=1
            in_offs_ptrs[length].append(start)
            out_offs_ptrs[length].append(position)
        else:
            assert length < 20000
            count[length]=1
            in_offs_ptrs[length]=[start]
            out_offs_ptrs[length]=[position]
        position+=length
    #import pdb;pdb.set_trace()
    assert len(count) < 7

    in_offs_ptr=[None]*6
    out_offs_ptr=[None]*6
    cnt = [0]*6
    sz = [0]*6
    for i,(l,c) in enumerate(count.items()):
        in_offs_ptr[i]=torch.tensor(in_offs_ptrs[l],dtype=torch.int32,device='npu')
        out_offs_ptr[i]=torch.tensor(out_offs_ptrs[l],dtype=torch.int32,device='npu')
        cnt[i]=c
        sz[i]=l

    output_tensor = torch.empty((n_rows, elements), dtype=input_tensor.dtype).npu()

    npu_triton_slice_cat_kernel_4_qianchuan[n_rows,1,1](
        input_tensor,output_tensor,
        in_offs_ptr[0],out_offs_ptr[0],
        in_offs_ptr[1],out_offs_ptr[1],
        in_offs_ptr[2],out_offs_ptr[2],
        in_offs_ptr[3],out_offs_ptr[3],
        in_offs_ptr[4],out_offs_ptr[4],
        in_offs_ptr[5],out_offs_ptr[5],
        stride=input_stride,
        output_xnumel=elements,
        cnt_1=cnt[0],
        cnt_4=cnt[1],
        cnt_8=cnt[2],
        cnt_16=cnt[3],
        cnt_32=cnt[4],
        cnt_64=cnt[5],
        sz_1=sz[0],
        sz_4=sz[1],
        sz_8=sz[2],
        sz_16=sz[3],
        sz_32=sz[4],
        sz_64=sz[5],
    )

    return output_tensor

#@fused_slice_cat.register_fake
@impl(npu_meta, "fused_slice_cat")
def fused_slice_cat_fake(
    input_tensor, int_list, elements, n_rows, input_stride
    
):
    # input_tensor: torch.Tensor,
    # slices: list,
    # n_rows: int,
    # input_stride: int,
       
#     input_tensor: torch.Tensor,
#     slices: list,
#     n_rows: int,
#     input_stride: int,
# ) -> torch.Tensor:
    
    # elements = 0    
    # for start,end in slices:
    #     length = end - start
    #     elements += length

    return torch.empty(n_rows, elements, device=input_tensor.device)
