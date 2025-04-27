import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import AutoMatchPattern


class ScaleDynamicQuantize(AutoMatchPattern):
    def rewriter(self, gm: fx.GraphModule, rule_name: str, node_map: dict):
        scale_node = node_map["scale"]
        quantize_node = node_map["dynamic_quantize"]

        if len(scale_node.users) != 1:
            return False

        with gm.graph.inserting_before(scale_node):
            scale_value_node = scale_node.args[1]
            if scale_value_node.meta["val"].dtype != torch.float:
                scale_value_node = gm.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(scale_value_node,),
                    kwargs={"dtype": torch.float},
                )
            if scale_node.target == torch.ops.aten.div.Tensor:
                scale_value_node = gm.graph.call_function(
                    torch.ops.aten.reciprocal.default, args=(scale_value_node,)
                )

            scale_dynamic_quantize = gm.graph.call_function(
                torch.ops.xpu_ops.scale_dynamic_quantize.default,
                args=(scale_node.args[0], scale_value_node),
            )

        quantize_node.replace_all_uses_with(scale_dynamic_quantize)
        gm.graph.erase_node(quantize_node)
        gm.graph.erase_node(scale_node)

        return True
