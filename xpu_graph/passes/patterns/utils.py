import torch
import torch.fx as fx

def expand_tensor(gm : fx.GraphModule, inp, src_node) -> fx.Node:
    scalar_tup = (int, float,)
    if type(inp) in scalar_tup:
        return gm.graph.call_function(
            torch.ops.aten.full.default,
            args=(
                src_node.meta['tensor_meta'].shape,
                inp
            ),
            kwargs={
                'dtype': src_node.meta['tensor_meta'].dtype,
                'memory_format': src_node.meta['tensor_meta'].memory_format,
                'device': src_node.meta['val'].device,
            }
        )
    else:
        return gm.graph.call_function(
            torch.ops.aten.expand.default,
            args=(
                inp,
                src_node.meta['tensor_meta'].shape,
            )
        )