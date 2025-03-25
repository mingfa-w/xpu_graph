from typing import Optional

import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from ...utils.check_ops import check_view


class FusedFFNReplacement(nn.Module):
    def forward(
        self,
        input,
        up_fc_weight,
        up_fc_bias,
        down_proj_weight,
        down_proj_bias,
        shape_param,
        act_mode,
    ):
        output = torch_mlu_ops.ffn(
            input,
            up_fc_weight,
            up_fc_bias,
            down_proj_weight,
            down_proj_bias,
            None,
            None,
            act_mode,
        )
        if shape_param:
            output = output.view(shape_param)
        return output


def _is_ffn(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    if (node.target != "mlu_tmo_fused_matmul_1_replacement") and (
        node.target != "mlu_tmo_fused_matmul_2_replacement"
    ):
        return False, []
    shape_param = None
    matmul_act_node = node.args[0]
    down_proj_input_shape = node.args[1]
    down_proj_weight = node.args[2]
    down_proj_weight_shape = node.args[3]
    down_proj_bias = node.args[5]
    shape_param = node.args[6]

    if (matmul_act_node.target != "mlu_tmo_fused_matmul_3_replacement") and (
        matmul_act_node.target != "mlu_tmo_fused_matmul_4_replacement"
    ):
        return False, []

    view_node = matmul_act_node.args[0]
    up_fc_input_shape = matmul_act_node.args[1]
    up_fc_weight = matmul_act_node.args[2]
    up_fc_weight_shape = matmul_act_node.args[3]
    up_fc_bias = matmul_act_node.args[5]
    act_mode = matmul_act_node.args[7]
    if not check_view(view_node):
        return False, []

    # is_trans = False
    if (
        up_fc_input_shape[1] == up_fc_weight_shape[1]
        and up_fc_weight_shape[0] == down_proj_weight_shape[1]
        and up_fc_input_shape[1] == down_proj_weight_shape[0]
    ):
        input = view_node.args[0]
        up_fc_weight = matmul_act_node.args[2]
        down_proj_weight = node.args[2]
        return True, [
            input,
            up_fc_weight,
            up_fc_bias,
            down_proj_weight,
            down_proj_bias,
            act_mode,
            shape_param,
        ]
    else:
        return False, []


class FusedFFN(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule("mlu_tmo_ffn_replacement", FusedFFNReplacement())
        for node in reversed(graph_module.graph.nodes):
            is_match, _is_ffn_param = _is_ffn(node)
            if is_match:
                (
                    input,
                    up_fc_weight,
                    up_fc_bias,
                    down_proj_weight,
                    down_proj_bias,
                    act_mode,
                    shape_param,
                ) = _is_ffn_param
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_tmo_ffn_replacement",
                        args=(
                            input,
                            up_fc_weight,
                            up_fc_bias,
                            down_proj_weight,
                            down_proj_bias,
                            shape_param,
                            act_mode,
                        ),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
