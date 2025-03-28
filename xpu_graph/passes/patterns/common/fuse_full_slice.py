import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.full_like import _is_full_like_node


#aten.slice(aten.full) -> aten.full
class FuseFullSlice(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor
        ]
        for node in candidates:
            input_node = node.args[0]
            dim = node.args[1]
            start = node.args[2]
            end = node.args[3]

            is_full, base_val = _is_full_like_node(input_node)
            if not is_full:
                continue

            original_shape = input_node.args[0]
            # Check: valid dim
            if not isinstance(dim, int) or not (0 <= dim < len(original_shape)):
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if not (0 <= start < end <= original_shape[dim]):
                continue

            # Build new shape after slice
            new_shape = list(original_shape)
            new_shape[dim] = end - start
            new_shape = tuple(new_shape)

            with gm.graph.inserting_after(input_node):
                new_node = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=(new_shape, base_val),
                    kwargs=input_node.kwargs,
                )

            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            changed = True

        if changed:
            gm.graph.lint()
            gm.recompile()

        return changed
