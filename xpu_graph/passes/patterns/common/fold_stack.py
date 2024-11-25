import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern

class FoldCat(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target == torch.ops.aten.stack.default]

        for stack in candidates:
            inps = stack.args[0]
            if len(inps) == 1:
                changed = True
                inp = inps[0]
                with gm.graph.inserting_before(stack):
                    view = gm.graph.call_function(
                        torch.ops.aten.view.default,
                        args=(
                            inp,
                            stack.meta['tensor_meta'].shape
                        )
                    )
                    stack.replace_all_uses_with(view)
                    gm.graph.erase_node(stack)


        gm.graph.lint()
        gm.recompile()
        return changed