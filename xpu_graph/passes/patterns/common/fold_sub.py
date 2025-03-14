import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.fx_utils import trace_and_inline, FxStage


class FoldSub0(Pattern):
    """
    Fold aten.sub(x, zero_like) -> x
    """
    _stages = [FxStage.inference, FxStage.pregrad]
    def process(self, gm: fx.GraphModule):
        changed = False
        sub_tup = (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target in sub_tup
        ]

        def _is_zero_like(inp) -> bool:
            scalar_tup = (
                int,
                float,
            )
            if type(inp) in scalar_tup and inp == 0:
                return True
            zero_like_tup = (
                torch.ops.aten.zeros_like.default,
                torch.ops.aten.zeros.default,
            )
            if (
                isinstance(inp, fx.Node)
                and inp.op == "call_function"
                and inp.target in zero_like_tup
            ):
                return True
            return False

        for sub in candidates:
            inp0 = sub.args[0]
            inp1 = sub.args[1]
            res = None
            is_match = False
            if _is_zero_like(inp1):
                is_match = True
                res = inp0

            if is_match:
                changed = True
                with gm.graph.inserting_before(sub):
                    from xpu_graph.passes.patterns.utils.expand_tensor import (
                        expand_tensor,
                    )

                    expand = expand_tensor(gm, res, sub)
                sub.replace_all_uses_with(expand)
                gm.graph.erase_node(sub)

        gm.graph.lint()
        gm.recompile()
        return changed
