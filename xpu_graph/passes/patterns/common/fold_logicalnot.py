import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern


def _is_full_like_node(node: fx.Node):
    if not (isinstance(node, fx.Node) and node.op == "call_function"):
        return False
    return node.target in (
        torch.ops.aten.full.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.zeros.default,
    )


def _get_value_and_shape(node: fx.Node):
    if node.target == torch.ops.aten.full.default:
        shape, value = node.args[0], node.args[1]
    elif node.target == torch.ops.aten.ones.default:
        shape, value = node.args[0], 1
    elif node.target == torch.ops.aten.zeros.default:
        shape, value = node.args[0], 0
    else:
        return None, None

    return value, tuple(shape)


class FoldLogicalNot(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        changed = False

        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.logical_not.default
        ]
        for node in candidates:
            input_node = node.args[0]
            if (
                isinstance(input_node, fx.Node)
                and input_node.op == "call_function"
                and input_node.target == torch.ops.aten.eq.Tensor
                and len(input_node.users) == 1
            ):
                x, y = input_node.args

                with gm.graph.inserting_before(node):
                    new_node = gm.graph.call_function(
                        torch.ops.aten.ne.Tensor, args=(x, y)
                    )

                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                changed = True
            elif _is_full_like_node(input_node):
                value, shape = _get_value_and_shape(input_node)
                if (value is None) or (shape is None):
                    continue
                value = bool(value)
                new_kwargs = dict(input_node.kwargs)
                new_kwargs["dtype"] = torch.bool
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.call_function(
                        torch.ops.aten.full.default,
                        args=(shape, not value),
                        kwargs=new_kwargs,
                    )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                changed = True

        if changed:
            gm.graph.lint()
            gm.recompile()

        return changed
