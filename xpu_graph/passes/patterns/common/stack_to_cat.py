from typing import Optional
import torch
from torch import nn, fx
from xpu_graph.passes.patterns.pattern import Pattern
from typing import Callable
from ..utils.check_ops import (
    check_stack_op,
)

def check_stack_inputs_shape(stack_node):
    inputs = stack_node.args[0]
    if not inputs:
        return False

    first_shape = None
    for node in inputs:
        if node is None:
            return False
        if not hasattr(node, "meta") or "tensor_meta" not in node.meta:
            return False

        shape = node.meta["tensor_meta"].shape
        if len(shape) != 2:
            return False

        if first_shape is None:
            first_shape = shape
        elif shape != first_shape:
            return False

    return True

def _is_stack(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    if not check_stack_op(node):
        return False, ()
    if len(node.args) != 1:
        return False, ()
    if not check_stack_inputs_shape(node):
        return False, ()

    inputs_list = node.args[0]
    shape = inputs_list[0].meta["tensor_meta"].shape
    input_nums = len(inputs_list)

    return True, (inputs_list, shape, input_nums)

class FusedStackToCat(Pattern):
    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False

        for node in reversed(graph_module.graph.nodes):
            is_match, params = _is_stack(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.call_function(
                        torch.ops.aten.cat.default,
                        args=(params[0], -1),
                        kwargs={},
                    )
                    reshape_node = graph_module.graph.call_function(
                        torch.ops.aten.view.default,
                        args=(cat_node, [params[1][0], params[2], params[1][1]]),
                        kwargs={},
                    )
                    permute_node = graph_module.graph.call_function(
                        torch.ops.aten.permute.default,
                        args=(reshape_node, [1, 0, 2]),
                        kwargs={},
                    )
                node.replace_all_uses_with(permute_node)
                graph_module.graph.erase_node(node)
                is_modified = True
        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
