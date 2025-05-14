import torch
import torch.fx as fx
import torch_mlu

from .triton_kernel.fused_slice import (
    fused_slice_low,
)

from .triton_kernel.fused_slice_cat import (
    fused_slice_cat,
)


class RMSNormModule(torch.nn.Module):
    def forward(self, inputs, weights, epsilon):
        import torch_mlu_ops

        return torch_mlu_ops.fused_rms_norm(
            inputs, None, weights, None, None, epsilon, False
        )


class FuseSliceModule(torch.nn.Module):
    def __init__(self, src_node, slices_index):
        super().__init__()
        device = torch.mlu.current_device()
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            self.slices_index = torch.tensor(
                slices_index, dtype=torch.int32, device="mlu:" + str(device)
            )

    def forward(self, input_tensor, slice_len):
        output = fused_slice_low(
            input_tensor,
            self.slices_index,
            slice_len,
            input_tensor.shape[0],
            input_tensor.stride(0),
        )
        return output.view(len(self.slices_index), input_tensor.shape[0], slice_len)


class FuseSliceCatSameInputModule(torch.nn.Module):
    def forward(self, input_tensor, slices):
        if len(input_tensor.shape) != 2:
            raise NotImplementedError("input must be 2d")
        indices = [i for start, end in slices for i in range(start, end)]
        rows, _ = input_tensor.shape
        indices_tensor = torch.tensor(
            indices, dtype=torch.int32, device=input_tensor.device
        )
        return fused_slice_cat(
            input_tensor,
            indices_tensor,
            rows,
            len(indices),
            input_tensor.stride(0),
        )


def get_structure_replacements():
    return {
        "FusedRMSNorm": RMSNormModule,
        "FusedSlice": FuseSliceModule,
        "FusedCatSlice": FuseSliceCatSameInputModule,
        "FusedSliceStackSum": FuseSliceCatSameInputModule,
        "FusedMultipleSliceCat": FuseSliceCatSameInputModule,
    }
