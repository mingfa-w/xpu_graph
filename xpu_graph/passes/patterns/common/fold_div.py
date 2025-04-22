import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern


class FoldDiv1(Pattern):
    """
    Fold aten.div(x, one_like) -> x
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        div_tup = (torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar)
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target in div_tup
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

        for div in candidates:
            inp0 = div.args[0]
            inp1 = div.args[1]
            target_val = None
            is_match = False
            if _is_one_like(inp1):
                is_match = True
                target_val = inp0

            if is_match:
                changed = True
                with gm.graph.inserting_before(div):
                    from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
                        get_binary_fold_result,
                    )

                    fold_res = get_binary_fold_result(gm, target_val, div.meta)
                div.replace_all_uses_with(fold_res)
                gm.graph.erase_node(div)

        gm.graph.lint()
        gm.recompile()
        return changed
