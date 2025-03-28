from typing import Optional, Union, Tuple

import torch
from torch import nn, fx
import torch.nn.functional as F
from typing import Callable, Optional, List
from xpu_graph.config import OptLevel

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from xpu_graph.fx_utils import FxStage

from ..utils.check_ops import (
    check_stack_op,
    check_sum_op,
)
from functools import reduce


def custom_add(tensor_list):
    return tensor_list[index]

class FusedStack(Pattern):
    _opt_level = OptLevel.level1
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        for node in reversed(graph_module.graph.nodes):
            is_sum = check_sum_op(node)
            if not is_sum:
                continue
            stack_node = node.args[0]
            axis = node.args[1]
            if axis != [0]:
                continue
            is_stack = check_stack_op(stack_node)
            if not is_stack:
                continue

            stack_inputs = stack_node.args[0]
            if not isinstance(stack_inputs, (list, tuple)):
                continue
            if len(stack_inputs) < 2:
                continue

            def make_add(a, b):
                return graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.add.Tensor,
                    args=(a, b),
                )

            with graph_module.graph.inserting_before(node):
                new_node = reduce(make_add, stack_inputs)
                new_node.name = f"replace_{node.name}"

            node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(node)
            graph_module.graph.erase_node(stack_node)
            changed = True

        graph_module.graph.lint()
        graph_module.recompile()
        return changed
