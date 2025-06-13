import torch
from torch import nn, fx
from torch.fx.node import Node
import torch.fx as fx
from typing import Optional, Tuple, List
import torch_npu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.config import OptLevel
import operator
from ...utils.check_ops import (
    check_add_op,
    check_norm_op,
    check_typecast_op
)

"""
def call_aten_kernel(input, residual, weight):
    new_residual_tensor = input.to(torch.float32) + residual.to(torch.float32)
    new_residual = (new_residual_tensor.to(dtype))
    eps = 1e-6
    x = new_residual
    rms = torch.sqrt((x.pow(2)).mean(-1, keepdim=True) + eps)  # 计算均方根
    x_norm = x / rms  # 标准化
    y_ref = weight * x_norm
    y_ref = y_ref.to(input.dtype)
    return y_ref, new_residual
"""

from .triton_kernel.fused_add_rmsnorm import fused_add_rmsnorm


class AddRmsnormOperation(nn.Module):
    def forward(self, input, residual, weight):
        return torch.ops.torch_npu_triton.fused_add_rmsnorm(input, residual, weight)


class FusedAddRmsnorm(Pattern):
    _opt_level = OptLevel.level2

    def _match_pattern(self, final_rmsnorm: Node) -> Optional[List[Node]]:
        # for qianwen model, add_RMSNorm_fuse_kernel
        res_flag, op_name = check_norm_op(final_rmsnorm)
        if (not res_flag) and (not op_name == "rms_norm"):
            return None

        to_node, weight_node = final_rmsnorm.args  # eps
        if not (check_typecast_op(to_node)):
            return None

        new_residual, new_residual_dty = to_node.args
        if (new_residual_dty != torch.bfloat16):
            return None
        if not (check_add_op(new_residual)):
            return None

        add1, add2 = new_residual.args
        if not (check_typecast_op(add1) and check_typecast_op(add2)):
            return None
        
        add1_inp, add1_dty = add1.args
        add2_inp, add2_dty = add2.args
        if (add1_dty != torch.float32 or add2_dty != torch.float32):
            return None

        return [
            add1_inp, 
            add2_inp,
            weight_node,
            final_rmsnorm,
            to_node
        ]


    def process(self, gm: fx.GraphModule):
        graph = gm.graph
        changed = False
        gm.add_submodule("npu_triton_fused_add_rmsnorm", AddRmsnormOperation())

        for node in reversed(list(graph.nodes)):
            matched_nodes = self._match_pattern(node)
            if not matched_nodes:
                continue
            changed = True

            (add1_inp, residual, weight_node,
            final_rmsnorm, to_node) = matched_nodes

            with graph.inserting_before(final_rmsnorm):
                # fused_to_node
                fused_rms_node = graph.call_module(
                    "npu_triton_fused_add_rmsnorm",
                    args=(
                        add1_inp, 
                        residual,
                        weight_node,
                    ),
                )
            
            with graph.inserting_after(fused_rms_node):
                fused_rms_node_out = graph.call_function(
                        operator.getitem,
                        args=(fused_rms_node, 0)
                )
                fused_to_node_out = graph.call_function(
                        operator.getitem,
                        args=(fused_rms_node, 1)
                )
            
            final_rmsnorm.replace_all_uses_with(fused_rms_node_out)
            to_node.replace_all_uses_with(fused_to_node_out)

            nodes_to_remove = [
                final_rmsnorm,
                to_node
            ]
            for n in nodes_to_remove:
                if len(n.users) == 0:
                    graph.erase_node(n)

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed