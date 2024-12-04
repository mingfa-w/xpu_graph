import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern

class FoldAdd0(Pattern):
    '''
    Fold aten.add(x, zero_like) -> x
    '''
    def process(self, gm: fx.GraphModule):
        changed = False
        add_tup = (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar,)
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target in add_tup]

        def _is_zero_like(inp) -> bool:
            scalar_tup = (int, float,)
            if type(inp) in scalar_tup and inp == 0:
                return True
            zero_like_tup = (torch.ops.aten.zeros_like.default, torch.ops.aten.zeros.default)
            if isinstance(inp, fx.Node) and inp.op == 'call_function' and inp.target in zero_like_tup:
                return True
            return False

        for add in candidates:
            inp0 = add.args[0]
            inp1 = add.args[1]
            res = None
            is_match = False
            if _is_zero_like(inp0):
                is_match = True
                res = inp1
            elif _is_zero_like(inp1):
                is_match = True
                res = inp0

            if is_match:
                changed = True
                with gm.graph.inserting_before(add):
                    from xpu_graph.passes.patterns.utils import expand_tensor
                    expand = expand_tensor(gm, res, add)
                add.replace_all_uses_with(res)
                gm.graph.erase_node(add)

        gm.graph.lint()
        gm.recompile()
        return changed