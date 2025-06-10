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
    check_sum_op,
    check_meta_2d,
    check_mul_op,
    get_shape,
)
from .combo_cat_utils import find_longest_same_shape_sequence, find_longest_same_param
from xpu_graph.fx_utils import FxStage
import operator

MAX_INT64 = 9223372036854775807

MINI_LEN = 3


def match_sum(val):
    if not check_sum_op(val):
        return False
    # if "mul_replacement" in val.name
    #    # shape infer
    #    return False
    users = val.users
    if len(users) > 2:
        return False
    if len(users) == 2:
        a, b = list(users.keys())
        if not (a.op == "output" or b.op == "output"):
            return False
    return True


class ComboSumCat(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and (node.target == torch.ops.aten.stack.default)
        ]

        for node in candidates:
            ori_cat_input = node.args[0]
            axis = 0  # node.args[1]
            if len(ori_cat_input) < MINI_LEN:
                continue
            start, end = match_sub_list(
                ori_cat_input,
                match_sum,
            )
            if end - start + 1 < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_shape_sequence(
                ori_cat_input, start, end, MINI_LEN, True
            )
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_param(
                ori_cat_input, best_start, best_end, MINI_LEN
            )
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue
            n_list = ori_cat_input[best_start : best_end + 1]
            sum_inputs = [n.args[0] for n in n_list]

            new_cat_input = ori_cat_input[:best_start]
            with graph_module.graph.inserting_before(node):
                fused_tensor = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(sum_inputs, axis),
                    name=node.name + "_sum_cat_replacement1",
                )
                sum_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.sum.dim_IntList,
                    args=(fused_tensor, n_list[0].args[1]),
                    name=node.name + "_sum_cat_replacement2",
                )
                new_shape = [len(n_list)] + list(get_shape(node))[1:]
                view_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.view.default,
                    args=(sum_node, new_shape),
                    kwargs={},
                )
                for i, sn in enumerate(n_list):
                    n = graph_module.graph.call_function(
                        operator.getitem,
                        args=(sum_node, i),
                        kwargs={},
                    )
                    sn.replace_all_uses_with(n)

            new_cat_input.append(sum_node)
            new_cat_input += ori_cat_input[best_end + 1 :]

            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, axis),
                        name=node.name + "_replacement",
                    )

            new_shape = [len(ori_cat_input)] + list(get_shape(node))[1:]
            with graph_module.graph.inserting_before(node):
                view_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.view.default,
                    args=(
                        cat_node if len(new_cat_input) > 1 else sum_node,
                        get_shape(node),
                    ),
                    kwargs={},
                )
            changed = True
            node.replace_all_uses_with(view_node)

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.cat.default
        ]
        for node in candidates:
            ori_cat_input = node.args[0]
            axis = node.args[1]
            if len(ori_cat_input) < MINI_LEN:
                continue
            start, end = match_sub_list(
                ori_cat_input,
                match_sum,
            )
            if end - start + 1 < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_shape_sequence(
                ori_cat_input, start, end, MINI_LEN, True
            )
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_param(
                ori_cat_input, best_start, best_end, MINI_LEN
            )
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue

            n_list = ori_cat_input[best_start : best_end + 1]
            sum_inputs = [n.args[0] for n in n_list]

            new_cat_input = ori_cat_input[:best_start]
            with graph_module.graph.inserting_before(node):
                # keepdim
                if len(n_list[0].args) == 3 and n_list[0].args[2] == True:
                    fused_tensor = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(sum_inputs, axis),
                        name=node.name + "_sum_cat_replacement1",
                    )
                    shape = list(get_shape(n_list[0].args[0]))
                    new_shape = shape[:axis] + [len(n_list)] + shape[axis:]
                    view_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.view.default,
                        args=(fused_tensor, new_shape),
                        kwargs={},
                    )
                    sum_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.sum.dim_IntList,
                        args=(view_node, [n_list[0].args[1][0] + 1]),
                        name=node.name + "_sum_cat_replacement2",
                    )
                else:
                    fused_tensor = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(sum_inputs, 0),
                        name=node.name + "_sum_cat_replacement1",
                    )
                    sum_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.sum.dim_IntList,
                        args=(fused_tensor, n_list[0].args[1]),
                        name=node.name + "_sum_cat_replacement2",
                    )
                sum_with_other_users = []
                for i, sn in enumerate(n_list):
                    if len(sn.users) != 1:  # 不只被cat使用
                        sum_with_other_users.append((i, sn))
                if sum_with_other_users != []:
                    single_sum_size = get_shape(n_list[0])[axis]
                    with graph_module.graph.inserting_after(sum_node):
                        split_results = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.split.Tensor,
                            args=(sum_node, single_sum_size, axis),
                            name=node.name + "_split_results",
                        )
                    with graph_module.graph.inserting_after(split_results):
                        for i, sn in sum_with_other_users:
                            split_part = graph_module.graph.call_function(
                                operator.getitem,
                                args=(split_results, i),
                                kwargs={},
                            )
                            sn.replace_all_uses_with(split_part)

            new_cat_input.append(sum_node)
            new_cat_input += ori_cat_input[best_end + 1 :]

            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, axis),
                        name=node.name + "_replacement",
                    )
                node.replace_all_uses_with(cat_node)
            else:
                node.replace_all_uses_with(sum_node)

            changed = True
        return changed
