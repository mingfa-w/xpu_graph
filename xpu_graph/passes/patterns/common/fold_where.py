import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.expand_tensor import expand_tensor
from xpu_graph.fx_utils import trace_and_inline, FxStage

class FoldWhere(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad]
    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target == torch.ops.aten.where.self]

        for where in candidates:
            inp = where.args[1]
            other = where.args[2]
            if inp == other:
                with gm.graph.inserting_before(where):
                    res = expand_tensor(gm, inp, where)
                    where.replace_all_uses_with(res)
                    gm.graph.erase_node(where)
                    changed = True
            elif isinstance(inp, fx.Node) and inp.op == 'call_function' and \
                isinstance(other, fx.Node) and other.op == 'call_function':
                if (inp.target == torch.ops.aten.ones_like.default and other.target == torch.ops.aten.ones_like.default) or \
                    (inp.target == torch.ops.aten.zeros_like.default and other.target == torch.ops.aten.zeros_like.default):
                    with gm.graph.inserting_before(where):
                        res = expand_tensor(gm, inp, where)
                        where.replace_all_uses_with(res)
                        gm.graph.erase_node(where)
                        changed = True

        gm.graph.lint()
        gm.recompile()
        return changed