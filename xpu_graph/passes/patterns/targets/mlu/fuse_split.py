import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.fx_utils import FxStage
from ...utils.check_ops import (
    check_meta_2d,
)
import operator

from .triton_kernel.fused_split import (
    fused_split,
)



'''
    %split : call_function[target=torch.ops.aten.split.Tensor](args = (%addmm_35, 380, 1), kwargs = {})
    %getitem_112 : [num_users=2] = call_function[target=operator.getitem](args = (%split, 0), kwargs = {})
    %getitem_113 : [num_users=2] = call_function[target=operator.getitem](args = (%split, 1), kwargs = {})
    %getitem_114 : [num_users=2] = call_function[target=operator.getitem](args = (%split, 2), kwargs = {})
    %getitem_115 : [num_users=2] = call_function[target=operator.getitem](args = (%split, 3), kwargs = {})
'''
class ReplaceSplitWithCustomModule(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        changed = False

        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.split.Tensor
        ]
        for node in candidates: 
            x = node.args[0]
            split_size = node.args[1]
            dim = node.args[2]
            if not check_meta_2d(x):
                continue
            if dim == 0:
                continue

            outputs = []
            index_nodes = []

            for user in list(node.users):
                if (
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and user.args[0] == node
                    and isinstance(user.args[1], int)
                ):
                    outputs.append(user.args[1])
                    index_nodes.append(user)

            if len(outputs) < 2:
                continue

            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    fused_split,
                    args=(x,split_size,dim)
                )
            for i, getitem_node in enumerate(index_nodes):
                with gm.graph.inserting_before(getitem_node):
                    out = gm.graph.call_function(
                        operator.getitem,
                        args=(new_node, outputs[i])
                    )
                    getitem_node.replace_all_uses_with(out)
                    gm.graph.erase_node(getitem_node)

            gm.graph.erase_node(node)
            changed = True

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed

