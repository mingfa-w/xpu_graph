import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern

class FoldDuplicate(Pattern):
    '''
    Fold node which duplicate computation
    '''
    def process(self, gm: fx.GraphModule):
        changed = False
        seen = {}
        for node in list(gm.graph.nodes):
            if node.op in ("call_function", "call_method"):
                if isinstance(node.target, str) and (node.target.endswith('_')):
                    continue
                key = (node.op, node.target, node.args, tuple(sorted(node.kwargs.items())) if node.kwargs else None)
                if key in seen:
                    changed = True
                    original_node = seen[key]
                    node.replace_all_uses_with(original_node)
                    gm.graph.erase_node(node)
                else:
                    seen[key] = node

        gm.graph.lint()
        gm.recompile()
        return changed
