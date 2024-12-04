import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import AutoMatchPattern

class Gelu(AutoMatchPattern):
    def rewriter(self, gm: fx.GraphModule, rule_name: str, node_map: dict) -> bool:
        assert len(node_map) == 5

        div_node = node_map['div']
        add_node = node_map['add']
        mul_node = node_map['mul']
        mul2_node = node_map['mul2']

        import math
        if div_node.args[1] != math.sqrt(2):
            return False
        if add_node.args[1] != 1 and add_node.args[0] != 1:
            return False
        if mul_node.args[1] != 0.5 and mul_node.args[0] != 0.5:
            return False

        with gm.graph.inserting_before(div_node):
            gelu_node = gm.graph.call_function(torch.ops.aten.gelu.default, args=(div_node.args[0],))

        mul2_node.replace_all_uses_with(gelu_node)

        gm.graph.lint()
        gm.recompile()
        return True