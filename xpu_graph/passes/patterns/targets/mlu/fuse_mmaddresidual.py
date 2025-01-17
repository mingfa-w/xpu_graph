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
    check_op,
    check_mm_op,
    check_mul_op,
    get_input_node,
    get_actual_node,
)


class FusedMMAddReplacement(nn.Module):
    def forward(
        self, inputs, weight, bias, residual ,alpha, beta
    ):
        weight = weight.transpose(0,1)
        output = torch_mlu_ops.attention_project(
            inputs,
            weight,
            bias,
            residual,
            alpha,
            beta,
        )
        return output


def check_fused_addmm(node: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_op(node, "fused_addmm"):
        return False
    return True

def _is_mm(node: fx.Node):
    mm_node = node
    mm_match, mm_input1, mm_input2 = check_mm_op(mm_node)
    if not mm_match:
        return False, ()
    mm_input1 = get_actual_node(mm_input1,0)
    return True, (mm_input1,mm_input2,None,None,1,1)

def _is_fuse_node(node: fx.Node, match_str: str):
    fuded_node = node
    if fuded_node.target == match_str:
        return True,(fuded_node.args[0],fuded_node.args[1],fuded_node.args[2],fuded_node.args[3],fuded_node.args[4],fuded_node.args[5])
    return False, None    

def _is_tmo_mm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    return _is_fuse_node(node, "fused_mmadd")

def _is_tmo_mm_add(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    return _is_fuse_node(node, "fused_mmadd1")

def _check_mul_op(node: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_mul_op(node):
        return False, None, None
    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    return True, arg1, arg2

def _check_add_op(node: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_add_op(node):
        return False, None, None
    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    return True, arg1, arg2
  
def _is_add_tmo_mm(node: fx.Node):
    add_node = node
    add_match, add_input1, add_input2 = _check_add_op(add_node)
    if not add_match:
        return False, ()
    tmo_mm_node = get_actual_node(add_input1,0)
    tmo_mm_match, tmo_mm_params= _is_tmo_mm(tmo_mm_node)
    if not tmo_mm_match:
        return False, ()
    return True, (tmo_mm_params[0],tmo_mm_params[1],add_input2,tmo_mm_params[3],tmo_mm_params[4],tmo_mm_params[5])

def _is_add1_tmo_mm(node: fx.Node):
    add_node = node
    add_match, add_input1, add_input2 = _check_add_op(add_node)
    if not add_match:
        return False, ()
    mul_match1, mul_input1_1, mul_input1_2 = _check_mul_op(add_input1)
    mul_match2, mul_input2_1, mul_input2_2 = _check_mul_op(add_input2)
    if not mul_match2:
        return False, ()
    residual = mul_input2_1
    beta = mul_input2_2
    if not mul_match1:
        return False, ()
    alpha = mul_input1_2
    tmo_mm_node = mul_input1_1
    tmo_mm_match, tmo_mm_params = _is_tmo_mm_add(tmo_mm_node)
    if not tmo_mm_match:
        return False, ()
    return True, (tmo_mm_params[0],tmo_mm_params[1],tmo_mm_params[2],residual,alpha,beta)

    

class FusedMMAdd(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule("fused_mmadd", FusedMMAddReplacement())

        for node in reversed(graph_module.graph.nodes):
            is_mm_match, mm_param = _is_mm(node)
            if is_mm_match:
                input, weight, bias, residual, alpha,beta = mm_param
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "fused_mmadd",
                        args=(
                            input,
                            weight,
                            bias,
                            residual,
                            alpha,
                            beta,
                        ),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        graph_module.add_submodule("fused_mmadd1", FusedMMAddReplacement())
        
        for node in reversed(graph_module.graph.nodes):
            is_add_tmo_mm_match, tmo_mm_param = _is_add_tmo_mm(node)
            if is_add_tmo_mm_match:
                input, weight, bias, residual, alpha,beta = tmo_mm_param
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "fused_mmadd1",
                        args=(
                            input,
                            weight,
                            bias,
                            residual,
                            alpha,
                            beta,
                        ),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True
        
        graph_module.add_submodule("fused_mmadd2", FusedMMAddReplacement())
        for node in reversed(graph_module.graph.nodes):
            is_add_tmo_mm_match_2, tmo_mm_param_2 = _is_add1_tmo_mm(node)
            if is_add_tmo_mm_match_2:
                input, weight, bias, residual, alpha,beta = tmo_mm_param_2
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "fused_mmadd2",
                        args=(
                            input,
                            weight,
                            bias,
                            residual,
                            alpha,
                            beta,
                        ),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True
        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified