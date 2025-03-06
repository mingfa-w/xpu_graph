import torch
import torch.fx as fx
import torch_npu


from .triton_kernel.fused_slice import (
    fused_slice_low,
)

from .triton_kernel.fused_slice_cat import (
    fused_slice_cat,
)

class LayerNormModule(torch.nn.Module):
    def forward(self, input, weight, bias, epsilon):
        return torch.nn.functional.layer_norm(
            input, input.shape[-1:], weight, bias, epsilon
        )


class FuseSliceModule(torch.nn.Module):
    def forward(self, input_tensor, slices_index, slice_len):
        #import pdb;pdb.set_trace()
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
        for start,end in slices:
            length = end - start
            elements += length
            elements = sum(end-start for start,end in slices)
        import itertools
        plain_slices = list(itertools.chain(*slices))

        rows, _ = input_tensor.shape
        return torch.ops.torch_npu_triton.fused_slice_cat(
            input_tensor,
            plain_slices, 
            elements,
            rows,
            input_tensor.stride(0),
        )
        
        


def get_structure_replacements():
    return {
        "FusedLayerNorm": LayerNormModule,
        "FusedSlice": FuseSliceModule,
        "FusedCatSlice": FuseSliceCatSameInputModule,
        "FusedMultipleSliceCat": FuseSliceCatSameInputModule,
    }
