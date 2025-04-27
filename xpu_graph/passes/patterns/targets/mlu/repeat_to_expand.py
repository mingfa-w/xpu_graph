from typing import Optional, Tuple, Union, List

import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.fx_utils import FxStage
from ...utils.check_ops import (
    get_shape,
    check_where_op,
    check_repeat_op,
)

TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node


"""
    sample_code:
    %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze, [1, 1, 256]), kwargs = {})
    %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%arg0_1, 1, %repeat), kwargs = {})
"""
class FusedGatherToCopy(Pattern):
    _opt_level = OptLevel.level1
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.gather.default
        ]
        for gather_node in candidates:
            repeat_node = gather_node.args[2]
            gather_dim = gather_node.args[1]
            if repeat_node.target != torch.ops.aten.repeat.default:
                continue
            if len(repeat_node.users) != 1:
                continue

            repeat_param = repeat_node.args[1]
            expand_param = [-1 if i == 1 else i for i in repeat_param]

            with graph_module.graph.inserting_before(repeat_node):
                new_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.expand.default,
                    args=(repeat_node.args[0], expand_param),
                    name=gather_node.name + "_replacement",
                )
            repeat_node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(repeat_node)
            is_modified = True
            graph_module.graph.lint()
            graph_module.recompile()
        return is_modified


"""
    sample_code:
    %repeat : [num_users=7] = call_function[target=torch.ops.aten.repeat.default](args = (%logical_or_13, [1, 32]), kwargs = {})
    %where_61 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%repeat, %add_40_replacement, %add_38_replacement), kwargs = {})
"""

def _is_repeat2expand(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[List]]:
    if not check_where_op(node):
        return False, None, []

    repeat_node = node.args[0]
    if not check_repeat_op(repeat_node):
        return False, None, []

    repeat_input = repeat_node.args[0]
    repeat_param = repeat_node.args[1]
    repeat_shape = repeat_input.meta["tensor_meta"].shape
    expand_param = list(repeat_param)
    if len(repeat_shape) == len(repeat_param):
        for i in range(len(repeat_param)):
            if repeat_param[i] != 1:
                if repeat_shape[i] != 1:
                    return False, None, []
            elif repeat_param[i] == 1:
                expand_param[i] = repeat_shape[i]
    else:
        for i in range(len(repeat_param)):
            if repeat_param[i] == 1:
                expand_param[i] = repeat_shape[i]
            else:
                return False, None, []

    return True, repeat_node, expand_param


class Repeat2Expand(Pattern):
    _opt_level = OptLevel.level1
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        is_modified = False

        for node in gm.graph.nodes:
            is_match, repeat_node, expand_param = _is_repeat2expand(node)
            if is_match:
                with gm.graph.inserting_before(repeat_node):
                    new_node = gm.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.expand.default,
                        args=(repeat_node.args[0], expand_param),
                        name=repeat_node.name + "_replacement",
                    )
                repeat_node.replace_all_uses_with(new_node)
                gm.graph.erase_node(repeat_node)
                is_modified = True

        gm.graph.lint()
        gm.recompile()
        return is_modified
