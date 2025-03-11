from typing import Optional

import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from ...utils.check_ops import (
    check_cat_op,
    check_where_op,
    check_zeros_op,
    check_slice_op
)
from .triton_kernel.fused_slice_where_cat import fuse_slice_where_cat


class FusedSliceWhereCatReplacement(nn.Module):
    def forward(
        self,
        where_input,
        slice_input,
        zeros_param,
        unmatched_nodes,
        cat_dim,
        slice_dim,
        slice_params
    ):
        processor_count = torch.mlu.get_device_properties(
            torch.mlu.current_device()
        ).multi_processor_count

        slice_param_tensor = torch.tensor(slice_params, dtype=torch.int32, device=where_input.device)
        slice_min = min(slice_params)
        slice_max = max(slice_params)
        slice_num = int(len(slice_params) / 2)
        cat_matched = fuse_slice_where_cat(where_input, slice_input, slice_param_tensor, slice_min, slice_max, zeros_param[1], processor_count, slice_num)

        if unmatched_nodes:
            unmatched_nodes = unmatched_nodes + [cat_matched]
            output = torch.cat(unmatched_nodes, dim=cat_dim)
            return output
        else:
            return cat_matched

def find_matching_nodes(cat_node):
    cat_inputs = cat_node.args[0]
    last_input = cat_inputs[-1]
    if not check_where_op(last_input):
        return False, ()
    if len(last_input.users) != 1:
        return False, ()
    matched_nodes = [last_input]
    
    if not hasattr(last_input, "args") or len(last_input.args) != 3:
        return False, ()

    condition, x, y = last_input.args[:3]
    is_zeros, zeros_param = check_zeros_op(x)
    if not is_zeros:
        return False, ()
    if not check_slice_op(y):
        return False, ()

    slice_input = y.args[0]
    slice_dim = y.args[1]
    slice_params = [y.args[3], y.args[2]]

    for node in reversed(cat_inputs[:-1]):
        if not hasattr(node, "args") or len(node.args) != 3:
            break

        condition_, x_, y_ = node.args[:3]
        if not check_slice_op(y_):
            break
        if len(node.users) != 1:
            break

        if condition_ == condition and x_ == x and y_.args[0] == y.args[0] and y_.args[1] == y.args[1]:
            matched_nodes.append(node)
            slice_params.append(y_.args[3])
            slice_params.append(y_.args[2])
        else:
            break

    if len(matched_nodes) < 2:
        return False, ()

    unmatched_nodes = [node for node in cat_inputs if node not in matched_nodes[::-1]]

    return True, (condition, slice_input, zeros_param, unmatched_nodes, slice_dim, slice_params[::-1])

def _is_slice_where_cat(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    is_cat, cat_dim = check_cat_op(node)
    if not is_cat:
        return False, (), ()
    is_match, param_tuple = find_matching_nodes(node)
    if not is_match:
        return False, (), ()

    where_input_shape = param_tuple[0].meta["tensor_meta"].shape
    slice_input_shape = param_tuple[1].meta["tensor_meta"].shape
    if len(where_input_shape) != 2 or len(slice_input_shape) != 2:
        return False, (), ()
    if where_input_shape[1] != 1:
        return False, (), ()
    if cat_dim != 1 and cat_dim != -1:
        return False, (), ()
    if param_tuple[4] != 1 and param_tuple[4] != -1:
        return False, (), ()

    return True, param_tuple, cat_dim

class FusedSliceWhereCat(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule("mlu_triton_slice_where_cat_replacement", FusedSliceWhereCatReplacement())
        for node in reversed(graph_module.graph.nodes):
            is_match, param_tuple, cat_dim = _is_slice_where_cat(node)
            if is_match:
                (
                    where_input,
                    slice_input,
                    zeros_param,
                    unmatched_nodes,
                    slice_dim,
                    slice_params
                ) = param_tuple

                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_triton_slice_where_cat_replacement",
                        args=(
                            where_input,
                            slice_input,
                            zeros_param,
                            unmatched_nodes,
                            cat_dim,
                            slice_dim,
                            slice_params
                        ),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
