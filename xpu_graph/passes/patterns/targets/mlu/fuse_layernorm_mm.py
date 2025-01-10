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


class FusedNormMMReplacement(nn.Module):
    def forward(self, input, normalized_shape, weight, bias, eps,q_weight,q_bias,k_weight=None,k_bias=None,v_weight=None,v_bias=None):
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


def _is_norm(node: fx.Node, match_str: str):
    layernorm_node = node
    is_norm, norm_type = check_norm_op(layernorm_node)
    
    if not is_norm:
        return False, ()
    
    if norm_type != match_str:
        return False, ()
    inputs = layernorm_node.args[0]
    normalized_shape = layernorm_node.args[1]
    weight = layernorm_node.args[2]
    bias = layernorm_node.args[3]
    eps = layernorm_node.args[4]

    # TODO(yhj): if len(add_node.users) > 1: return residual
    return True, (inputs, normalized_shape, weight, bias, eps)


def _is_layernorm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    if not check_getitem_op(node):
        return False, ()
    return _is_norm(node.args[0],"layer_norm")

def _is_rmsnorm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    return _is_norm(node, "rms_norm")

def _check_add_op(node: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_add_op(node):
        return False, None, None
    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    return True, arg1, arg2


def _is_add_mm_norm(node: fx.Node):
    add_node = node
    add_match, add_input1, add_input2 = _check_add_op(add_node)
    if not add_match:
        mm_node = node
        mm_match, mm_input1, mm_input2 = check_mm_op(mm_node)
        if not mm_match:
            return False, ()
        norm_node = get_actual_node(mm_input1,0)
        norm_match, norm_params= _is_layernorm(norm_node)
        if not norm_match:
            return False, ()
        inputs = norm_params[0]
        residual = norm_params[1]
        weight = norm_params[2]
        bias = norm_params[3]
        eps = norm_params[4]
        q_weight = mm_input2
        q_bias = add_input2
        return True, (inputs,residual,weight,bias,None,eps,q_weight,q_bias)
    else:
        
        mm_node = get_actual_node(add_input1,0)
        mm_match, mm_input1, mm_input2 = check_mm_op(mm_node)
        if not mm_match:
            return False, ()
        norm_node = get_actual_node(mm_input1,0)
        norm_match, norm_params= _is_layernorm(norm_node)
        if not norm_match:
            return False, ()
        inputs = norm_params[0]
        residual = norm_params[1]
        weight = norm_params[2]
        bias = norm_params[3]
        eps = norm_params[4]
        q_weight = mm_input2
        q_bias = add_input2
        return True, (inputs,residual,weight,bias,None,eps,q_weight,q_bias)

class FusedNormMM(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule(
            "mlu_tmo_fused_norm_mm_replacement", FusedNormMMReplacement()
        )
        for node in reversed(graph_module.graph.nodes):
            is_match, params = _is_add_mm_norm(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_tmo_fused_norm_mm_replacement",
                        args=(
                            params[0],
                            params[1],
                            params[2],
                            params[3],
                            params[5],
                            params[6],
                            params[7],
                            ),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True        
        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified