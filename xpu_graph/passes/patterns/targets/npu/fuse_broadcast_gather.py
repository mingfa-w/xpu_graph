import torch
from torch import nn, fx
from torch.fx.node import Node
import torch.fx as fx
from typing import Optional, Tuple, List
import torch_npu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.config import OptLevel
from ...utils.check_ops import (
    check_expand_op,
    check_gather_op,
)

import os
from torch._ops import ops
xpu_ops_lib = "/host/wxue/xpu_ops/src/build_out/lib/libxpu_ops.so"

if not os.path.exists(xpu_ops_lib):
    print(xpu_ops_lib)
    raise RuntimeError('can not find xpu ops lib')

torch.ops.load_library(xpu_ops_lib)
torch.classes.load_library(xpu_ops_lib)


@torch.library.register_fake("xpu_ops::broadcast_gather")
def broadcast_gather(input, index, dim):
    # 这里实现伪实现的逻辑,简单地返回输入的形状信息
    output_shape = list(index.shape)
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)


"""
view_11: "f16[11, 12, 256, 512]" = torch.ops.aten.view.default(bmm_1, [11, 12, 256, 512]);  bmm_1 = None
expand_1: "i64[11, 12, 256, 256]" = torch.ops.aten.expand.default(clamp_max, [11, 12, -1, -1]);
gather: "f16[11, 12, 256, 256]" = torch.ops.aten.gather.default(view_11, -1, expand_1);
"""

# class ExpandGatherOperation(nn.Module):
#     def forward(self, view_11, expand_1):
#         return torch.ops.xpu_ops.boardcast_gather(view_11, expand_1, -1)

class FusedExpandGather(Pattern):
    _opt_level = OptLevel.level2

    def _match_pattern(self, final_gather: Node) -> Optional[List[Node]]:
        if not (check_gather_op(final_gather)):
            return None

        view_node, gather_dim, index_node = final_gather.args

        if gather_dim != -1:
            return None

        if not (check_expand_op(index_node)):
            return None

        clamp_node, _ = index_node.args
        return [
            view_node, clamp_node, final_gather
        ]


    # but there was no fake impl or Meta kernel registered
    def process(self, gm: fx.GraphModule):
        graph = gm.graph
        changed = False

        for node in reversed(list(graph.nodes)):
            matched_nodes = self._match_pattern(node)
            if not matched_nodes:
                continue
            changed = True

            (view_node, clamp_node, final_gather) = matched_nodes

            #breakpoint()
            index_data = clamp_node.args[0]

            with graph.inserting_before(final_gather):
                fused_node = graph.call_function(
                    torch.ops.xpu_ops.broadcast_gather,
                    args = (view_node, index_data, -1),
                )
            final_gather.replace_all_uses_with(fused_node)
            changed = True

            nodes_to_remove = [view_node, clamp_node]
            for n in nodes_to_remove:
                if len(n.users) == 0:
                    graph.erase_node(n)

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed