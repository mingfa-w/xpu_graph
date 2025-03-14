import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.fx_utils import trace_and_inline, FxStage

MAX_INT64 = 9223372036854775807


class FoldSlice(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad]
    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor
        ]

        for node in candidates:
            if (node.args[2] == 0) & (node.args[3] == MAX_INT64):
                src_node = node.args[0]
                node.replace_all_uses_with(src_node)
                gm.graph.erase_node(node)
                changed = True

        gm.graph.lint()
        gm.recompile()
        return changed
