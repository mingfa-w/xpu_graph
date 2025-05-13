import torch                                                                                                                                                                                                                                                                    
import torch_mlu                                                                                                                                                                                                                                                                
import triton                                                                                                                                                                                                                                                                   
import triton.language as tl                                                                                                                                                                                                                                                    
from . import libentry                                                                                                                                                                                                                                                          
from .get_mlu_devinfo import get_device_properties                                                                                                                                                                                                                              
from typing import List, Tuple                                                                                                                                                                                                                                                  

@torch.library.custom_op("torch_mlu_triton::fused_emb_cat", mutates_args=())
def fused_emb_cat(
    input_tensor: torch.Tensor,
    input_offsets: List[int],
    input_dims: List[int],
    output_ids: List[int],
    output_offsets: List[int],
    output_dims: List[int],
) -> List[torch.Tensor]:
    input_dims = torch.tensor(
        input_dims, device=input_tensor.device, dtype=torch.int32
    )
    input_offsets = torch.tensor(
        input_offsets, device=input_tensor.device, dtype=torch.int32
    )
    output_dims = torch.tensor(
        output_dims, device=input_tensor.device, dtype=torch.int32
    )
    output_offsets = torch.tensor(
        output_offsets, device=input_tensor.device, dtype=torch.int32
    )
    output_dims_cpu = torch.tensor(output_dims, device="cpu", dtype=torch.int32)
    output_ids = torch.tensor(
        output_ids, device=input_tensor.device, dtype=torch.int32
    )
    '''
    print(input_tensor.shape)
    print(input_offsets)
    print(input_dims)
    print(output_ids)
    print(output_offsets)
    print(output_dims)
    '''

    output = torch.ops.torch_mlu.emb_concat(
        input_tensor,
        input_offsets,
        input_dims,
        output_ids,
        output_offsets,
        output_dims,
        output_dims_cpu,
    )
    return output


@fused_emb_cat.register_fake
def fused_emb_cat_fake(
    input_tensor: torch.Tensor,
    input_offsets: List[int],
    input_dims: List[int],
    output_ids: List[int],
    output_offsets: List[int],
    output_dims: List[int],
    ):
    batch = input_tensor.shape[0]
    outputs = []
    for dim in output_dims:
        outputs.append(torch.empty(batch, dim, device=input_tensor.device, dtype=input_tensor.dtype))
    return outputs
