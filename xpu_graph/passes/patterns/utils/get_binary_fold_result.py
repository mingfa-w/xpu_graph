import torch
import torch.fx as fx
from typing import Union, Dict, Any


def get_binary_fold_result(
    gm: fx.GraphModule, inp: Union[int, float, fx.Node], target_meta: Dict[str, Any]
) -> fx.Node:
    scalar_tup = (
        int,
        float,
    )
    assert type(inp) in scalar_tup or isinstance(
        inp, fx.Node
    ), "get_binary_fold_result input error"

    if type(inp) in scalar_tup:
        return gm.graph.call_function(
            torch.ops.aten.full.default,
            args=(target_meta["tensor_meta"].shape, inp),
            kwargs={
                "dtype": target_meta["tensor_meta"].dtype,
                "device": target_meta["val"].device,
            },
        )
    else:
        expand = gm.graph.call_function(
            torch.ops.aten.expand.default,
            args=(
                inp,
                target_meta["tensor_meta"].shape,
            ),
        )
        copy = gm.graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(expand,),
            kwargs={
                "memory_format": torch.contiguous_format,
            },
        )
        return copy
