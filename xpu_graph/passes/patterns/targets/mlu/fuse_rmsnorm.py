# from typing import Optional

# import torch
# from torch import nn, fx
# import torch_mlu
# import torch_mlu_ops

# from xpu_graph.passes.patterns.pattern import Pattern
# from xpu_graph.utils import logger
# from ...utils.check_ops import (
#     check_add_op,
#     check_mul_op,
#     check_pow_op,
#     check_mean_op,
#     check_rsqrt_op,
#     get_input_node,
# )


# def _is_rmsnorm(
#     node: fx.Node,
# ) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
#     if not check_mul_op(node):
#         return (False, None, None, None)

#     weight_mul_node = get_input_node(node, 1)
#     if not check_mul_op(weight_mul_node):
#         return (False, None, None, None)

#     rsqrt_node = get_input_node(weight_mul_node, 1)
#     if not check_rsqrt_op(rsqrt_node):
#         return (False, None, None, None)

#     add_node = get_input_node(rsqrt_node, 0)
#     if not check_add_op(add_node):
#         return (False, None, None, None)

#     mean_node = get_input_node(add_node, 0)
#     if not check_mean_op(mean_node):
#         return (False, None, None, None)

#     pow_node = get_input_node(mean_node, 0)
#     if not check_pow_op(pow_node):
#         return (False, None, None, None)

#     return (True, add_node, mean_node, pow_node)


# class RMSNormModule(nn.Module):
#     def forward(self, inputs, weights, epsilon):
#         return torch_mlu_ops.fused_rms_norm(
#             inputs, None, weights, None, None, epsilon, False
#         )


# class FusedRMSNorm(Pattern):
#     def process(self, graph_module: fx.GraphModule) -> bool:
#         is_modified = False
#         graph_module.add_submodule("rms_norm_op", RMSNormModule())

#         for node in reversed(graph_module.graph.nodes):
#             matched, add_node, mean_node, pow_node = _is_rmsnorm(node)
#             if not matched:
#                 continue

#             input_node = get_input_node(pow_node, 0)
#             weight_node = get_input_node(node, 0)
#             if input_node is None or weight_node is None:
#                 continue

#             epsilon = (
#                 add_node.args[1]
#                 if len(add_node.args) > 1 and isinstance(add_node.args[1], (float, int))
#                 else 1e-6
#             )

#             with graph_module.graph.inserting_before(node):
#                 rms_norm_node = graph_module.graph.call_module(
#                     "rms_norm_op", args=(input_node, weight_node, epsilon)
#                 )

#             node.replace_all_uses_with(rms_norm_node)
#             is_modified = True

#         graph_module.graph.lint()
#         graph_module.recompile()
#         return is_modified
