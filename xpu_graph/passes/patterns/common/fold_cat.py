import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern

class FoldCat(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target == torch.ops.aten.cat.default]

        for cat in candidates:
            inps = cat.args[0]
            if len(inps) == 1:
                changed = True

                inp = inps[0]
                cat.replace_all_uses_with(inp)
                gm.graph.erase_node(cat)

        gm.graph.lint()
        gm.recompile()
        return changed