from torch import nn, fx
import torch
import torch_mlu
from typing import Optional
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.match_sub_list import match_sub_list
from ...utils.check_ops import (
    check_cat_op,
    check_where_op,
    check_slice_op,
    check_meta_2d,
    check_zeros_op,
)
from .combo_cat_utils import find_longest_same_shape_sequence, find_longest_same_input
from xpu_graph.fx_utils import FxStage


MINI_LEN = 3


def match_slice_where(val):
    if len(val.users) != 1:
        return False
    if not check_where_op(val):
        return False
    condition, x, y = val.args[:3]
    if not check_zeros_op(x):
        return False
    if not check_slice_op(y):
        return False
    return True


class ComboSliceWhereCat(Pattern):
    _opt_level = OptLevel.level1
    """
    slice_1 -> where_1 ----\
    slice_2 -> where_2 ------> cat -> output
    slice_3 -> where_3 ----/
    other   ---------------/
    ---->
    slice_1 ----\
    slice_2 -----> cat -> where -> cat -> output  
    slice_3 ----/              /
    other   ------------------/
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and (node.target == torch.ops.aten.cat.default)
        ]

        for node in candidates:
            ori_cat_input = node.args[0]
            axis = node.args[1]
            if len(ori_cat_input) < MINI_LEN:
                continue
            start, end = match_sub_list(
                ori_cat_input,
                match_slice_where,
            )
            if end - start + 1 < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_input(
                ori_cat_input, start, end, MINI_LEN
            )
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue
            n_list = ori_cat_input[best_start : best_end + 1]
            where_inputs = [n.args[2] for n in n_list]
            with graph_module.graph.inserting_before(node):
                slice_cat_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(where_inputs, axis),
                    name=node.name + "_combo_slice_where_cat_replacement1",
                )
                # where_op = torch.zeros_like(slice_cat_node)
                zeros_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.zeros_like.default,
                    args=(slice_cat_node,),
                    name=node.name + "_combo_slice_where_cat_zeros",
                    kwargs=dict(n_list[0].args[1].kwargs),
                )
                where_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.where.self,
                    args=(n_list[0].args[0], zeros_node, slice_cat_node),
                    name=node.name + "_combo_slice_where_cat_replacement2",
                )

            new_cat_input = (
                ori_cat_input[:best_start]
                + [where_node]
                + ori_cat_input[best_end + 1 :]
            )
            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, axis),
                        name=node.name + "_combo_slice_where_cat_replacement3",
                    )
                node.replace_all_uses_with(cat_node)
            else:
                node.replace_all_uses_with(where_node)
            changed = True
        if changed:
            print(graph_module.graph)
        return changed
