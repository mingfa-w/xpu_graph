import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import is_node_escaped


class FoldClone(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad, FxStage.forward, FxStage.backward]

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.clone.default
        ]

        for clone in candidates:
            inp = clone.args[0]
            if is_node_escaped(inp):
                continue
            org_memoryformat = inp.meta["tensor_meta"].memory_format
            target_memoryformat = (
                clone.kwargs["memory_format"]
                if "memory_format" in clone.kwargs
                else org_memoryformat
            )
            if org_memoryformat == target_memoryformat:
                changed = True
                clone.replace_all_uses_with(inp)
                gm.graph.erase_node(clone)

        gm.graph.lint()
        gm.recompile()
        return changed
