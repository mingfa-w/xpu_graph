import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.full_like import _is_full_like_node


def _get_shape(node: fx.Node):
    shape = node.args[0]
    return tuple(shape)


# %logical_or : [num_users=1] = call_function[target=torch.ops.aten.logical_or.default](args = (%ones_default, %ones_default_1), kwargs = {})
class FuseLogicalFullTwoSide1(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        changed = False

        for node in list(gm.graph.nodes):
            enable_node = [
                torch.ops.aten.logical_or.default,
                torch.ops.aten.logical_and.default,
            ]

            if not (node.op == "call_function" and node.target in enable_node):
                continue

            lhs, rhs = node.args

            is_full, val1 = _is_full_like_node(lhs)
            if not is_full:
                continue
            is_full, val2 = _is_full_like_node(rhs)
            if not is_full:
                continue

            shape1 = _get_shape(lhs)
            shape2 = _get_shape(rhs)

            if shape1 != shape2 or val1 is None or val2 is None:
                continue

            val1 = bool(val1)
            val2 = bool(val2)
            if node.target == torch.ops.aten.logical_or.default:
                result_val = bool(val1 or val2)
            else:
                result_val = bool(val1 and val2)

            new_kwargs = dict(lhs.kwargs)
            new_kwargs["dtype"] = torch.bool
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=(shape1, result_val),
                    kwargs=new_kwargs,
                )

            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            changed = True

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed


class FuseLogicalFullTwoSide2(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        changed = False

        for node in gm.graph.nodes:
            enable_node = [
                torch.ops.aten.eq.Tensor,
                torch.ops.aten.ne.Tensor,
                torch.ops.aten.eq.Scalar,
                torch.ops.aten.ne.Scalar,
            ]

            if not (node.op == "call_function" and node.target in enable_node):
                continue

            lhs, rhs = node.args

            is_full, val1 = _is_full_like_node(lhs)
            if not is_full:
                continue
            is_full, val2 = _is_full_like_node(rhs)
            if not is_full:
                continue

            shape1 = _get_shape(lhs)
            shape2 = _get_shape(rhs)

            if shape1 != shape2 or val1 is None or val2 is None:
                continue

            if (node.target == torch.ops.aten.eq.Tensor) or (
                node.target == torch.ops.aten.eq.Scalar
            ):
                result_val = val1 == val2
            else:
                result_val = val1 != val2

            new_kwargs = dict(lhs.kwargs)
            new_kwargs["dtype"] = torch.bool
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=(shape1, result_val),
                    kwargs=new_kwargs,
                )

            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            changed = True

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed
