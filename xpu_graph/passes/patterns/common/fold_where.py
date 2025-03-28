import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.expand_tensor import expand_tensor
from xpu_graph.passes.patterns.utils.full_like import _is_full_like_node_v


class FoldWhere(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad, FxStage.forward, FxStage.backward]

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.where.self
        ]

        for where in candidates:
            cond = where.args[0]
            inp = where.args[1]
            other = where.args[2]

            # torch.where(full/zeros/ones, x, y) -> x or y
            if _is_full_like_node_v(cond, 1):
                with gm.graph.inserting_before(where):
                    res = expand_tensor(gm, inp, where)
                    where.replace_all_uses_with(res)
                    gm.graph.erase_node(where)
                    changed = True
                    continue
            elif _is_full_like_node_v(cond, 0):
                with gm.graph.inserting_before(where):
                    res = expand_tensor(gm, other, where)
                    where.replace_all_uses_with(res)
                    gm.graph.erase_node(where)
                    changed = True
                    continue
            # torch.where(cond, a, a) -> broad_cast(a)
            elif inp == other:
                with gm.graph.inserting_before(where):
                    res = expand_tensor(gm, inp, where)
                    where.replace_all_uses_with(res)
                    gm.graph.erase_node(where)
                    changed = True
                    continue
            # torch.where(cond, zeros, zeros) -> broad_cast(zeros)
            # torch.where(cond, ones, ones) -> broad_cast(ones)
            elif (_is_full_like_node_v(inp, 1) and _is_full_like_node_v(other, 1)) or (
                _is_full_like_node_v(inp, 0) and _is_full_like_node_v(other, 0)
            ):
                with gm.graph.inserting_before(where):
                    res = expand_tensor(gm, inp, where)
                    where.replace_all_uses_with(res)
                    gm.graph.erase_node(where)
                    changed = True
                    continue
            # torch.where(cond, ones, zeros) -> cond
            elif _is_full_like_node_v(inp, 1) and _is_full_like_node_v(other, 0):
                with gm.graph.inserting_before(where):
                    res = expand_tensor(gm, cond, where)
                    where.replace_all_uses_with(res)
                    gm.graph.erase_node(where)
                    changed = True
                    continue
            elif _is_full_like_node_v(inp, 1):
                where.args = (cond, cond, where.args[2])
                changed = True
                continue

            elif _is_full_like_node_v(other, 0):
                where.args = (cond, where.args[1], cond)
                changed = True
                continue

        gm.graph.lint()
        gm.recompile()
        return changed
