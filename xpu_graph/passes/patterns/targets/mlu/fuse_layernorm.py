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
    check_layernorm_op,
    check_view,
    check_getitem_op,
)


class FusedLayerNormReplacement(nn.Module):
    def forward(self, input, residual, weight, bias, residual_bias, eps):
        if bias is None:
            bias = torch.empty_like(weight)
        output = torch_mlu_ops.fused_layer_norm(
            input,
            residual,
            weight,
            bias,
            None,
            eps,
            False,
        )
        return output


def _is_add_layernorm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    if not check_getitem_op(node):
        return (False, None, None, None, None, None, None)

    layernorm_node = node.args[0]
    if not check_layernorm_op(layernorm_node):
        return (False, None, None, None, None, None, None)

    add_node = layernorm_node.args[0]
    if not check_add_op(add_node):
        return (False, None, None, None, None, None, None)
    # normalized_shape = layernorm_node.args[1]
    weight = layernorm_node.args[2]
    bias = layernorm_node.args[3]
    eps = layernorm_node.args[4]

    if len(add_node.users) == 1:
        inputs = add_node.args[0]
        residual = add_node.args[1]
        inputs_shape = inputs.meta["tensor_meta"].shape
        residual_shape = residual.meta["tensor_meta"].shape
        return (True, inputs, residual, weight, bias, None, eps)
    else:
        # [TODO] if len(add_node.users) > 1: return residual
        return (False, None, None, None, None, None, None)


class FusedLayerNorm(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False

        graph_module.add_submodule(
            "mlu_tmo_fused_layernorm_replacement", FusedLayerNormReplacement()
        )
        for node in reversed(graph_module.graph.nodes):
            (
                is_match,
                inputs,
                residual,
                weight,
                bias,
                residual_bias,
                eps,
            ) = _is_add_layernorm(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_tmo_fused_layernorm_replacement",
                        args=(inputs, residual, weight, bias, residual_bias, eps),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
