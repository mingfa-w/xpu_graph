from typing import Optional
import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    check_sum_op,
    check_squeeze_op,
    check_mul_op,
    check_slice_op,
    check_cat_op,
    check_meta_2d,
)
from torch.fx.subgraph_rewriter import replace_pattern
from xpu_graph.fx_utils import FxStage

from .triton_kernel.fused_mul_sum import fused_mul_sum

MAX_INT64 = 9223372036854775807

class FusedMulSumReplacement(nn.Module):
    def forward(self, mul0, mul1, slice_l=None, slice_r=None):
        result = fused_mul_sum(mul0, mul1, slice_l)
        return result

def _is_mul_sum(
    node: fx.Node
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node]]:
    if (not check_sum_op(node)
        or len(node.args) != 3
        or node.args[1] != [1]
        or node.args[2] != True
        or len(node.args[0].meta["tensor_meta"].shape) != 2
    ):
        return False, (), ()

    squeeze_node = node.args[0]
    if not check_squeeze_op(squeeze_node) or len(squeeze_node.users) > 1:
        return False, (), ()
    squeeze_dim = squeeze_node.args[1]
    if squeeze_dim != [2] and squeeze_dim != 2:
        return False, (), ()

    mul_node = squeeze_node.args[0]
    if not check_mul_op(mul_node) or len(mul_node.users) > 1:
        return False, (), ()

    slice_node = mul_node.args[0]
    _is_slice = False
    slice_r = 0
    slice_params = []
    if check_slice_op(slice_node):
        if slice_node.args[1] == 2 and len(slice_node.users) == 1:
            _is_slice = True
            slice_input = slice_node.args[0]
            slice_r = slice_node.args[3]
            if slice_r == MAX_INT64:
                slice_r = slice_input.meta["tensor_meta"].shape[2]
            elif slice_r < 0:
                slice_r = slice_input.meta["tensor_meta"].shape[2] + slice_r
            slice_params.append(slice_node.args[2])
            slice_params.append(slice_r)

    inputs = ()
    if _is_slice:
        if slice_params[-1] - slice_params[-2] != 1:
            return False, (), ()
        inputs = (slice_node.args[0], mul_node.args[1], )
    else:
        inputs = mul_node.args

    return True, inputs, tuple(slice_params)

class FusedMulSum(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        print(graph_module.graph)
        graph_module.add_submodule("mlu_triton_mul_sum_replacement", FusedMulSumReplacement())

        for node in reversed(graph_module.graph.nodes):
            is_match, inputs, slice_params = _is_mul_sum(node)
            if is_match:
                inputs_total = inputs + slice_params
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_triton_mul_sum_replacement",
                        args=(inputs_total)
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                changed = True

        graph_module.graph.lint()
        graph_module.recompile()
        return changed
