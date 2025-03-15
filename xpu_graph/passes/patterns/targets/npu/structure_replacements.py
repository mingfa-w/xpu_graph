import torch
import torch.fx as fx
import torch_npu

from .triton_kernel.fused_slice import (
    fused_slice_low,
)

from .triton_kernel.fused_slice_cat import (
    fused_slice_cat,
)

from .triton_kernel.shortcut_gather import (
    shortcut_gather,
)

class LayerNormModule(torch.nn.Module):
    def forward(self, input, weight, bias, epsilon):
        if weight is not None:
            if (input.dtype != weight.dtype):
                weight = weight.to(input.dtype)
        if bias is not None:
            if (input.dtype != bias.dtype):
                bias = bias.to(input.dtype)
        return torch.nn.functional.layer_norm(
            input, input.shape[-1:], weight, bias, epsilon
        )


class FuseSliceModule(torch.nn.Module):
    def forward(self, input_tensor, slices_index, slice_len):
        if len(input_tensor.shape) != 2:
            raise NotImplementedError("input must be 2d")
        if slice_len > input_tensor.shape[-1]:
            raise NotImplementedError(
                f"inputshape {input_tensor.shape} don't support slice_len:{slice_len}"
            )
        slices_index = torch.tensor(
            slices_index, dtype=torch.int32, device=input_tensor.device
        )
        output = torch.ops.torch_npu_triton.fused_slice_low(
            input_tensor,
            slices_index,
            slice_len,
            input_tensor.shape[0],
            input_tensor.stride(0),
        )
        return output

class FuseSliceCatSameInputModule(torch.nn.Module):
    def forward(self, input_tensor, slices):
        #import pdb;pdb.set_trace()
        if len(input_tensor.shape) != 2:
            raise NotImplementedError("input must be 2d")
        
        elements = 0
        rows, _ = input_tensor.shape
        count = {}
        in_offs_ptrs={}
        out_offs_ptrs={}
        position=0 

        for i, (start, end) in enumerate(slices):
            length = end - start
            elements += length
            if length in count:
                count[length]+=1
                in_offs_ptrs[length].append(start)
                out_offs_ptrs[length].append(position)
            else:
                assert length < 20000 and "slice length greater than UB !"
                count[length]=1
                in_offs_ptrs[length]=[start]
                out_offs_ptrs[length]=[position]
            position+=length

        assert len(count) < 7 and "too much category of length"

        in_offs_ptr=[None]*6
        out_offs_ptr=[None]*6
        cnt = [0]*6
        sz = [0]*6
        for i,(l,c) in enumerate(count.items()):
            in_offs_ptr[i]=torch.tensor(in_offs_ptrs[l],dtype=torch.int32,device='npu')
            out_offs_ptr[i]=torch.tensor(out_offs_ptrs[l],dtype=torch.int32,device='npu')
            cnt[i]=c
            sz[i]=l

        return torch.ops.torch_npu_triton.fused_slice_cat(
            input_tensor,rows,
            in_offs_ptr[0],out_offs_ptr[0],
            in_offs_ptr[1],out_offs_ptr[1],
            in_offs_ptr[2],out_offs_ptr[2],
            in_offs_ptr[3],out_offs_ptr[3],
            in_offs_ptr[4],out_offs_ptr[4],
            in_offs_ptr[5],out_offs_ptr[5],
            stride=input_tensor.stride(0),
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
        
class ShortCutGatherModule(torch.nn.Module):
    def forward(self, input_tensor, dim, prefix_len):
        if not (1 < len(input_tensor.shape) < 4):
            raise NotImplementedError("input must be 2d or 3d")
        return torch.ops.torch_npu_triton.shortcut_gather(
            input_tensor,
            dim,
            prefix_len,
        )


def get_structure_replacements():
    return {
        "FusedLayerNorm": LayerNormModule,
        "FusedSlice": FuseSliceModule,
        "FusedCatSlice": FuseSliceCatSameInputModule,
        "FusedMultipleSliceCat": FuseSliceCatSameInputModule,
        "SingleContinuousGather": ShortCutGatherModule,
    } 
