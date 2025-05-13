import torch
import torch.fx as fx
import torch_mlu
from .triton_kernel.get_mlu_devinfo import get_device_properties

from .triton_kernel.fused_slice import (
    fused_slice_low,
)

from .triton_kernel.fused_slice_cat import (
    fused_slice_cat,
)

from .triton_kernel.fused_slice_v2 import (
    fused_slice_low_v2,
)

from .triton_kernel.fused_sum2d import (
    fused_sum_2d,
)

from .triton_kernel.fused_emb_cat import (
    fused_emb_cat,
)


class RMSNormModule(torch.nn.Module):
    def forward(self, inputs, weights, epsilon):
        import torch_mlu_ops

        return torch_mlu_ops.fused_rms_norm(
            inputs, None, weights, None, None, epsilon, False
        )


class FuseSliceModule(torch.nn.Module):
    def forward(self, input_tensor, slices_index, slice_len):
        slices_index = torch.tensor(
            slices_index, dtype=torch.int32, device=input_tensor.device
        )
        output = fused_slice_low(
            input_tensor,
            slices_index,
            slice_len,
            input_tensor.shape[0],
            input_tensor.stride(0),
        )
        return output.view(len(slices_index), input_tensor.shape[0], slice_len)


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


class FuseSliceCatSameInputModule_v2(torch.nn.Module):
    def forward(self, input_tensor, many_slices):
        if 0:
            if len(input_tensor.shape) != 2:
                raise NotImplementedError("input must be 2d")
            indices = []
            total_output = []
            slices_index = [0]
            for slices in many_slices:
                sum_ = 0
                for slice_ in slices:
                    start, end = slice_
                    indices += range(start, end)
                    sum_ += end - start
                slices_index.append(sum_ + slices_index[-1])
                total_output.append(sum_)
            indices_tensor = torch.tensor(
                indices, dtype=torch.int32, device=input_tensor.device
            )

            output_total = fused_slice_cat(
                input_tensor,
                indices_tensor,
                input_tensor.shape[0],
                len(indices),
                input_tensor.stride(0),
            )

            slices_index = torch.tensor(
                slices_index[:-1], dtype=torch.int32, device=input_tensor.device
            )
            outputs = fused_slice_low_v2(
                output_total,
                slices_index,
                total_output,
            )
            return output
        else:
            num_output = 0
            output_ids = []
            input_dims = []
            output_dims = []
            output_offsets = []
            input_offsets = []
            for slices in many_slices:
                sum_ = 0
                for slice_ in slices:
                    start, end = slice_
                    slice_len = end - start
                    input_offsets.append(start)
                    input_dims.append(slice_len)
                    output_ids.append(num_output)
                    output_offsets.append(sum_)
                    sum_ += slice_len
                output_dims.append(sum_)
                num_output += 1

            return fused_emb_cat(
                input_tensor,
                input_offsets,
                input_dims,
                output_ids,
                output_offsets,
                output_dims,
            )


class ComboSumModule(torch.nn.Module):
    def forward(self, input_list, dim):
        return fused_sum_2d(input_list, dim[0])
        if len(input_list) < 2:
            return [torch.sum(input, dim=dim) for input in input_list]

        if dim != [1] and dim != [2]:
            return [torch.sum(input, dim=dim) for input in input_list]

        return fused_sum_2d(input_list, dim[0])


def get_structure_replacements():
    return {
        "FusedRMSNorm": RMSNormModule,
        "FusedSlice": FuseSliceModule,
        "FusedCatSlice": FuseSliceCatSameInputModule,
        "FusedSliceStackSum": FuseSliceCatSameInputModule,
        "FusedMultipleSliceCat": FuseSliceCatSameInputModule_v2,
        "ComboSum2d": ComboSumModule,
    }
