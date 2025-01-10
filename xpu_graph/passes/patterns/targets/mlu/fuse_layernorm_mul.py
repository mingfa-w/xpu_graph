from typing import Optional

import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    check_add_op,
    check_norm_op,
    check_view,
    check_getitem_op,
    check_mm_op,
    get_input_node,
    get_actual_node,
    check_trans_op,
)


class FusedNormMulReplacement(nn.Module):
    def forward(self, input, residual, weight, bias, residual_bias, eps,q_weight,q_bias,k_weight,k_bias,v_weight,v_bias):
        if bias is None:
            bias = torch.zeros_like(weight)
        q_weight = q_weight.transpose(1,0)
        output = torch_mlu_ops.fused_norm_attention_project(
            input,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            weight,
            bias,
            eps,
            "nthc",
            1,
            False
        )
        return output


def _is_norm(node: fx.Node):
    layernorm_node = node
    is_norm, norm_type = check_norm_op(layernorm_node)
    
    if not is_norm:
        return False, ()
    inputs = layernorm_node.args[0]
    weight = layernorm_node.args[2]
    bias = layernorm_node.args[3]
    eps = layernorm_node.args[4]

    # TODO(yhj): if len(add_node.users) > 1: return residual
    return True, (inputs, None, weight, bias, None, eps)


def _is_layernorm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    if not check_getitem_op(node):
        return False, ()
    return _is_norm(node.args[0])


def _check_add_op(node: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_add_op(node):
        return False, None, None
    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    return True, arg1, arg2

def _is_mm(
    node: fx.Node,
):
    match_, input1, input2 = check_mm_op(node)
    if match_:
        input1 = get_actual_node(node, 0)
        input2 = get_actual_node(node, 1)
        return True, (input1, input2)
    return match_, ()

def _is_add(
    node: fx.Node,
):
    match_, input1, input2 = _check_add_op(node)
    if match_:
        input1 = get_actual_node(node, 0)
        input2 = get_actual_node(node, 1)
        return True, (input1, input2)
    return match_, ()
def _check_trans_op(node: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_trans_op(node):
        return False, None
    arg1 = get_input_node(node, 0)
    return True, arg1

def _is_trans(
    node: fx.Node,
):
    match_, args = _check_trans_op(node)
    if match_:
        args = get_actual_node(node, 0)
        return True, (args,)
    return match_, ()

class FusedNormMul(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule(
            "mlu_tmo_fused_norm_mm_replacement", FusedNormMulReplacement()
        )
        is_norm_match = False
        is_mul_match_0 = False
        is_add_match_0 = False
        for node in reversed(graph_module.graph.nodes):
            is_norm_match, norm_params = _is_layernorm(node)
            if is_norm_match:
                break
        for node in reversed(graph_module.graph.nodes):
            if node.name == "mm":
                is_mul_match_0, mul_params_0 = _is_mm(node)
                break
        for node in reversed(graph_module.graph.nodes):
            if node.name == "add":
                is_add_match_0, add_params_0 = _is_add(node)
                if is_add_match_0:
                    headnode = node
                    break
        
        if is_norm_match and is_mul_match_0 and is_add_match_0:
            k_weight = None
            k_bias = None
            v_weight = None
            v_bias = None
            with graph_module.graph.inserting_before(headnode):
                new_node = graph_module.graph.call_module(
                    "mlu_tmo_fused_norm_mm_replacement",
                    args=(
                        norm_params[0],
                        norm_params[1],
                        norm_params[2],
                        norm_params[3],
                        norm_params[4],
                        norm_params[5],
                        mul_params_0[1],
                        add_params_0[1],
                        k_weight,
                        k_bias,
                        v_weight,
                        v_bias,
                        ),
                )
            node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(node)
            is_modified = True    
        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified