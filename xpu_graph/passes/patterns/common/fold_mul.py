import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.fx_utils import trace_and_inline, FxStage

class FoldMul1(Pattern):
    """
    Fold aten.mul(x, one_like) -> x
    """
    _stages = [FxStage.inference, FxStage.pregrad]
    def process(self, gm: fx.GraphModule):
        changed = False
        mul_tup = (
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul.Scalar,
        )
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target in mul_tup
        ]

        def _is_one_like(inp) -> bool:
            scalar_tup = (
                int,
                float,
            )
            if type(inp) in scalar_tup and inp == 1:
                return True
            one_like_tup = (
                torch.ops.aten.ones_like.default,
                torch.ops.aten.ones.default,
            )
            if (
                isinstance(inp, fx.Node)
                and inp.op == "call_function"
                and inp.target in one_like_tup
            ):
                return True
            return False

        for mul in candidates:
            inp0 = mul.args[0]
            inp1 = mul.args[1]
            res = None
            is_match = False
            if _is_one_like(inp0):
                is_match = True
                res = inp1
            elif _is_one_like(inp1):
                is_match = True
                res = inp0

            if is_match:
                changed = True
                with gm.graph.inserting_before(mul):
                    from xpu_graph.passes.patterns.utils.expand_tensor import (
                        expand_tensor,
                    )

                    expand = expand_tensor(gm, res, mul)
                mul.replace_all_uses_with(expand)
                gm.graph.erase_node(mul)

        gm.graph.lint()
        gm.recompile()
        return changed
