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
    check_where_op,
    check_full_op,
    check_ones_op,
    check_zeros_op,
    check_eq_op,
)

class MergeCatReplacement(nn.Module):
    def forward(self, cond, a, b):
        return cond*a + ~cond*b


'''
    %arg0_1 : [num_users=5] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %zeros_default : [num_users=1] = call_function[target=torch.ops.aten.zeros.default](args = ([2048, 1],), kwargs = {dtype: torch.int64, device: mlu:0, pin_memory: False})
    %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, 1), kwargs = {})
    %ones_default : [num_users=1] = call_function[target=torch.ops.aten.ones.default](args = ([2048, 1],), kwargs = {dtype: torch.int64, device: mlu:0, pin_memory: False})
    %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %ones_default, %zeros_default), kwargs = {})
    %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, 2), kwargs = {})
    %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ((2048, 1), 2), kwargs = {dtype: torch.int64, device: mlu:0})
    %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default, %where), kwargs = {})
    %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, 3), kwargs = {})
    %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ((2048, 1), 3), kwargs = {dtype: torch.int64, device: mlu:0})
    %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_2, %full_default_1, %where_1), kwargs = {})
    %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, 9998), kwargs = {})
    %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ((2048, 1), 4), kwargs = {dtype: torch.int64, device: mlu:0})
    %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_3, %full_default_2, %where_2), kwargs = {})
    %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg0_1, 9999), kwargs = {})
    %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ((2048, 1), 5), kwargs = {dtype: torch.int64, device: mlu:0})
    %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_4, %full_default_3, %where_3), kwargs = {})
'''
class FusedFillWhereLeft(Pattern):
    _opt_level = OptLevel.level1
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        graph_module.add_submodule("tmp_where", MergeCatReplacement())
        changed = False
        candidates = [
            node
            for node in graph_module.graph.nodes
            if check_where_op(node) 
        ]
        for node in candidates: 
            is_where = check_where_op(node)
            if not is_where:
                continue

            flag = 0
            full_node_left = node.args[1]
            if check_full_op(full_node_left):
                full_node_left = full_node_left.args[1]
                flag = 1
            elif check_ones_op(full_node_left):
                full_node_left = 1 
                flag = 1
            elif check_zeros_op(full_node_left):
                full_node_left = 0 
                flag = 1

            full_node_right = node.args[2]
            if check_full_op(full_node_right):
                full_node_right = full_node_right.args[1]
                flag = 1
            elif check_ones_op(full_node_right):
                full_node_right = 1 
                flag = 1
            elif check_zeros_op(full_node_right):
                full_node_right = 0 
                flag = 1

            if flag == 0:
                continue

            cond_node = node.args[0]

            with graph_module.graph.inserting_before(node):
                new_node = graph_module.graph.call_module(
                    "tmp_where", args=(cond_node, full_node_left, full_node_right)
                )
            node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(node)
            changed = True

        graph_module.graph.lint()
        graph_module.recompile()
        print(graph_module.graph)
        return changed

