import torch
from torch import nn, fx
from xpu_graph.passes.patterns.pattern import Pattern
from typing import Callable
from ..utils.check_ops import (
    check_cat_op,
    check_slice_op,
    check_stack_op,
    check_meta_2d,
)

MAX_INT64 = 9223372036854775807


class MergeCatReplacement(nn.Module):
    def forward(self, input_tensor_list, cat_axis=0):
        return torch.cat(
            [
                (
                    input_tensor
                    if len(input_tensor.shape) == 3
                    else input_tensor.unsqueeze(0)
                )
                for input_tensor in input_tensor_list
            ],
            axis=0,
        )


class ExpandTransReplacement(nn.Module):
    def forward(self, input_tensor, dim):
        return input_tensor.reshape(input_tensor.shape[0], dim, -1).transpose(0, 1)


def validate_slice_operation(n_list):
    if len(n_list) < 2:
        return False, None, None
    slice_input = []
    slice_param = []
    slice_axis = []
    for n in n_list:
        slice_input.append(n.args[0])
        slice_axis.append(n.args[1])
        right = n.args[3]
        if right == MAX_INT64:
            right = slice_input[0].meta["tensor_meta"].shape[-1]
        elif right < 0:
            right = slice_input[0].meta["tensor_meta"].shape[-1] - (-right)
        slice_param.append((n.args[2], right))
    if slice_input.count(slice_input[0]) != len(slice_input):
        return False, None, None
    if slice_axis.count(1) != len(slice_axis):
        return False, None, None
    return True, slice_input[0], slice_param


def extract_slice_info(nodes):
    slice_input = []
    slice_axis = []
    slice_param = []

    for node in nodes:
        if not check_slice_op(node):
            return False, [], [], []
        slice_input.append(node.args[0])
        slice_axis.append(node.args[1])
        slice_param.append((node.args[2], node.args[3]))

    return True, slice_input, slice_axis, slice_param


def insert_fuse_slice(graph_module, node, src_node, slice_param):
    with graph_module.graph.inserting_before(node):
        slice_node = graph_module.graph.call_module(
            "fuse_slice_cat",
            args=(src_node, slice_param),
        )
    return slice_node


def match_sub_list(lst):
    max_len = 0
    current_len = 0
    start_index = -1

    best_start = -1
    best_end = -1

    for i, val in enumerate(lst):
        if check_slice_op(val) and check_meta_2d(val):
            if current_len == 0:
                start_index = i
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                best_start = start_index
                best_end = i
        else:
            current_len = 0
    return best_start, best_end


def fuse_mixed_ops_and_catstack(graph_module: fx.GraphModule):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_cat, cat_axis = check_cat_op(node)
        if (not check_stack_op(node)) and (not is_cat):
            continue
        if is_cat:
            if not check_meta_2d(node):
                continue
            if cat_axis == 0:
                continue
        ori_cat_input = node.args[0]
        start, end = match_sub_list(ori_cat_input)
        n_list = node.args[0][start : end + 1]
        is_slice, src_node, slice_param = validate_slice_operation(n_list)
        if not is_slice:
            continue

        # mark these slice node bacause this pattern will cause side effects.
        for n in n_list:
            n.meta['changed_by_fused_slice_cat'] = True

        new_cat_input = ori_cat_input[:start]

        slice_node = insert_fuse_slice(graph_module, node, src_node, slice_param)

        if is_cat:
            new_cat_input.append(slice_node)
        else:
            with graph_module.graph.inserting_before(node):
                stack_node = graph_module.graph.call_module(
                    "expand_transpose_module",
                    args=(slice_node, len(slice_param)),
                )
            new_cat_input.append(stack_node)

        new_cat_input += ori_cat_input[end + 1 :]
        if is_cat:
            with graph_module.graph.inserting_before(node):
                cat_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(new_cat_input, -1),
                    name=node.name + "_replacement",
                )
        else:
            with graph_module.graph.inserting_before(node):
                cat_node = graph_module.graph.call_module(
                    "reshape_cat_module", args=(new_cat_input, -1)
                )
        node.replace_all_uses_with(cat_node)
        slice_nodes = node.args[0]
        for slice_node in slice_nodes:
            if len(slice_node.users) == 0:
                graph_module.graph.erase_node(slice_node)
        graph_module.graph.erase_node(node)
        changed = True

    return changed


class FusedCatSlice(Pattern):
    def __init__(self, target_mod: torch.nn.Module):
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule):
        changed = False
        graph_module.add_submodule(
            "fuse_slice_cat",
            self.target_mod(),
        )
        graph_module.add_submodule("expand_transpose_module", ExpandTransReplacement())
        graph_module.add_submodule("reshape_cat_module", MergeCatReplacement())

        # slice & cat, the inputs of cat are mixed with slice and other ops.
        changed = changed | fuse_mixed_ops_and_catstack(graph_module)
        # graph_module.graph.lint()
        # graph_module.recompile()
        return changed
