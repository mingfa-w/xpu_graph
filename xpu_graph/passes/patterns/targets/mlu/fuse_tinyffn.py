from typing import Optional

import torch
from torch import nn, fx
import torch_mlu

from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from .triton_kernel.fused_tinyffn import fuse_tinyffn


class FusedTinyFFNReplacement(nn.Module):
    def forward(
        self,
        tinyffn_input,
        up_fc_weight,
        up_fc_bias,
        down_proj_weight,
        down_proj_bias,
        act_mode_up,
        act_mode_down,
        is_transb_up,
        is_transb_down,
        is_up_bias,
        is_down_bias,
        is_up_act,
        is_down_act,
    ):
        if not tinyffn_input.is_contiguous():
            tinyffn_input = tinyffn_input.contiguous()
        if not up_fc_weight.is_contiguous():
            up_fc_weight = up_fc_weight.contiguous()
        if not down_proj_weight.is_contiguous():
            down_proj_weight = down_proj_weight.contiguous()
        output = fuse_tinyffn(
            tinyffn_input,
            up_fc_weight,
            up_fc_bias,
            down_proj_weight,
            down_proj_bias,
            #act_mode_up,
            #act_mode_down,
            is_up_bias,
            is_down_bias,
            is_up_act,
            is_down_act,
        )
        return output


def _is_tinyffn(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    if (node.target != "mlu_tmo_fused_matmul_replacement") and (
        node.target != "mlu_tmo_fused_matmul_add_replacement"
    ):
        return False, []

    matmul_act_node = node.args[0]
    down_proj_input_shape = node.args[1]
    down_proj_weight = node.args[2]
    down_proj_weight_shape = node.args[3]
    is_transb_down = node.args[4]
    down_proj_bias = node.args[5]
    act_mode_down = node.args[6]

    if (matmul_act_node.target != "mlu_tmo_fused_matmul_replacement") and (
        matmul_act_node.target != "mlu_tmo_fused_matmul_add_replacement") and (
        matmul_act_node.target != "mlu_tmo_fused_matmul_act_replacement") and (
        matmul_act_node.target != "mlu_tmo_fused_matmul_add_act_replacement"
    ):
        return False, []

    if len(matmul_act_node.users) > 1:
        return False, []

    tinyffn_input = matmul_act_node.args[0]
    up_fc_input_shape = matmul_act_node.args[1]
    up_fc_weight = matmul_act_node.args[2]
    up_fc_weight_shape = matmul_act_node.args[3]
    is_transb_up = matmul_act_node.args[4]
    up_fc_bias = matmul_act_node.args[5]
    act_mode_up = matmul_act_node.args[6]

    M, K0 = up_fc_input_shape
    K1, N1 = up_fc_weight_shape
    N2, O2 = down_proj_weight_shape

    if N1 != N2:
        return False, []

    #[TODO] remove
    if K0 == 820 and N1 == 128 and O2 == 128:
        return False, []
    if K0 == 356 and N1 == 512 and O2 == 64:
        return False, []
    if K0 == 640 and N1 == 256 and O2 == 128:
        return False, []

    size_of_dtype = 2
    input_dtype = tinyffn_input.meta['val'].dtype
    if input_dtype == torch.float32:
        size_of_dtype = 4
    if (K1 * N1 + N2 * O2 + N1 + O2) > (512 / size_of_dtype * 1024):
        return False, []

    is_up_bias, is_down_bias = False, False
    if up_fc_bias is not None:
        is_up_bias = True
    if down_proj_bias is not None:
        is_down_bias = True

    is_up_act, is_down_act = True, True
    if act_mode_up == "none":
        is_up_act = False
    elif act_mode_up == "relu":
        is_up_act = True
    else:
        return False, []

    if act_mode_down == "none":
        is_down_act = False
    elif act_mode_down == "relu":
        is_down_act = True
    else:
        return False, []


    if is_transb_up or is_transb_down:
        return False, []

    return True, [
        tinyffn_input,
        up_fc_weight,
        up_fc_bias,
        down_proj_weight,
        down_proj_bias,
        act_mode_up,
        act_mode_down,
        is_transb_up,
        is_transb_down,
        is_up_bias,
        is_down_bias,
        is_up_act,
        is_down_act,
    ]

class FusedTinyFFN(Pattern):
    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        #print(graph_module.graph)
        graph_module.add_submodule("mlu_triton_tinyffn_replacement", FusedTinyFFNReplacement())
        for node in reversed(graph_module.graph.nodes):
            is_match, tinyffn_param = _is_tinyffn(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_triton_tinyffn_replacement",
                        args=(tuple(tinyffn_param)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified
