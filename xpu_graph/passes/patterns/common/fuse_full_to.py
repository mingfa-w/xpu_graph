import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.full_like import _is_full_like_node

#aten.to(aten.full(dtype=torch.float32), dtype=int64)=>aten.full(dtype=torch.int64)
class FuseFullTo(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad, FxStage.forward, FxStage.backward]

    def process(self, gm: fx.GraphModule):
        changed = False

        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.to.dtype
        ]
        for node in candidates:
            input_node = node.args[0]
            dtype = node.args[1]
            is_full, base_val = _is_full_like_node(input_node)
            if not is_full:
                continue

            new_kwargs = dict(input_node.kwargs)
            new_kwargs["dtype"] = dtype

            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    input_node.target,
                    args=input_node.args,
                    kwargs=new_kwargs,
                )

            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            changed = True

        if changed:
            gm.graph.lint()
            gm.recompile()

        return changed
