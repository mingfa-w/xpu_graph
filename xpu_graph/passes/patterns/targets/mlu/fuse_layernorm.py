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
)


class FusedNormReplacement(nn.Module):
    def forward(self, input, residual, weight, bias, residual_bias, eps, norm_type):
        if bias is None:
            bias = torch.zeros_like(weight)
        if norm_type == "layer_norm":
            output = torch_mlu_ops.fused_layer_norm(
                input,
                residual,
                weight,
                bias,
                None,
                eps,
                False,
            )
        else:
            output = torch_mlu_ops.fused_rms_norm(
                input,
                residual,
                weight,
                bias,
                None,
                eps,
                False,
            )
        return output


def _is_add_norm(node: fx.Node, match_str: str):
    layernorm_node = node
    is_norm, norm_type = check_norm_op(layernorm_node)
    if not is_norm:
        return False, ()

    if norm_type != match_str:
        return False, ()

    add_node = layernorm_node.args[0]
    if not check_add_op(add_node):
        return False, ()
    # normalized_shape = layernorm_node.args[1]
    weight = layernorm_node.args[2]
    bias = layernorm_node.args[3]
    eps = layernorm_node.args[4]

    # TODO(yhj): if len(add_node.users) > 1: return residual
    if len(add_node.users) != 1:
        return False, ()

    inputs = add_node.args[0]
    residual = add_node.args[1]
    return True, (inputs, residual, weight, bias, None, eps)


def _is_add_layernorm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    if not check_getitem_op(node):
        return False, ()
    return _is_add_norm(node.args[0], "layer_norm")


def _is_add_rmsnorm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    return _is_add_norm(node, "rms_norm")


class FusedNorm(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False

        graph_module.add_submodule(
            "mlu_tmo_fused_norm_replacement", FusedNormReplacement()
        )

        for node in reversed(graph_module.graph.nodes):
            is_match, params = _is_add_layernorm(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_tmo_fused_norm_replacement",
                        args=(params + ("layer_norm",)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        for node in reversed(graph_module.graph.nodes):
            is_match, params = _is_add_rmsnorm(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_tmo_fused_norm_replacement",
                        args=(params + ("rms_norm",)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
