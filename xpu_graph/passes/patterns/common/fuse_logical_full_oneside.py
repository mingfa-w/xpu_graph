import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.full_like import _is_full_like_node


class FuseLogicalFullOneSide(Pattern):
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

            const_val = None
            tensor_node = None

            if isinstance(rhs, (int, float, bool)) and lhs.op == "call_function":
                tensor_node, const_val = lhs, rhs
            elif isinstance(lhs, (int, float, bool)) and rhs.op == "call_function":
                tensor_node, const_val = rhs, lhs
            else:
                continue

            is_full, base_val = _is_full_like_node(tensor_node)
            if not is_full:
                continue
            base_val = float(base_val)

            shape = tensor_node.args[0]
            if not isinstance(shape, (tuple, list)):
                continue

            if (tensor_node == torch.ops.aten.eq.Tensor) or (
                tensor_node == torch.ops.aten.eq.Scalar
            ):
                result_val = base_val == const_val
            else:  # ne
                result_val = base_val != const_val

            new_kwargs = dict(tensor_node.kwargs)
            new_kwargs["dtype"] = torch.bool
            with gm.graph.inserting_after(tensor_node):
                new_node = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=(shape, result_val),
                    kwargs=new_kwargs,
                )
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            changed = True

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed
