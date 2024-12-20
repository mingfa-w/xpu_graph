import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    check_add_op,
    check_bmm_op,
    check_softmax_op,
    get_input_node,
)


class FlashAttentionReplacement(nn.Module):
    def forward(self, query, key, value, scale_params, attention_mask):
        batch_size, num_heads, sequence_len, head_dim = query.shape
        key_sequence_len = key.shape[-2]

        query = query.reshape(-1, sequence_len, head_dim).transpose(0, 1)
        key = key.reshape(-1, key_sequence_len, head_dim).transpose(0, 1)
        value = value.reshape(-1, key_sequence_len, head_dim).transpose(0, 1)

        scale_factor, is_division = scale_params
        softmax_scale = 1.0 / scale_factor if is_division else scale_factor

        if attention_mask is not None:
            attention_mask = torch.broadcast_to(
                attention_mask, (batch_size, num_heads, sequence_len, key_sequence_len)
            ).contiguous()

        output = torch_mlu_ops.flash_attention(
            query,
            key,
            value,
            None,
            torch.tensor([0, sequence_len], dtype=torch.int32, device="mlu"),
            torch.tensor([0, key_sequence_len], dtype=torch.int32, device="mlu"),
            None,
            attention_mask,
            sequence_len,
            key_sequence_len,
            softmax_scale,
            False,
        )

        return (
            output.reshape(-1, sequence_len, num_heads, head_dim)
            .transpose(1, 2)
            .reshape(query.shape)
        )


class FlashAttentionWithTranspose(nn.Module):
    def __init__(self):
        super().__init__()
        self.flash_attention = FlashAttentionReplacement()

    def forward(self, query, key_transposed, value, scale_params, attention_mask):
        key = key_transposed.transpose(-1, -2)
        return self.flash_attention(query, key, value, scale_params, attention_mask)


def validate_transpose_operation(key_transpose):
    if not isinstance(key_transpose, fx.Node) or key_transpose.op != "call_function":
        return False

    if key_transpose.target != torch.ops.aten.transpose.int:
        return False

    dim1, dim2 = key_transpose.args[1:]
    valid_dimensions = [(-2, -1), (-1, -2), (1, 2), (2, 3)]
    return (dim1, dim2) in valid_dimensions


def validate_operation_counts(nodes):
    operation_counts = {"add": 0, "div": 0, "mul": 0}
    allowed_operations = {
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.clone.default,
        torch.ops.aten._to_copy.default,
    }

    for node in nodes:
        if node.target == torch.ops.aten.add.Tensor:
            operation_counts["add"] += 1
        elif node.target == torch.ops.aten.div.Tensor:
            operation_counts["div"] += 1
        elif node.target == torch.ops.aten.mul.Tensor:
            operation_counts["mul"] += 1
        elif node.target not in allowed_operations:
            logger.warning(f"Flash attention pass: Unsupported operation {node.target}")
            return False, 0, 0

    if operation_counts["add"] > 1:
        logger.warning("Flash attention pass: Too many add operations")
        return False, 0, 0

    scale_ops = operation_counts["div"] + operation_counts["mul"]
    if scale_ops > 1:
        logger.warning("Flash attention pass: Too many scaling operations")
        return False, 0, 0

    return True, operation_counts["add"], scale_ops


def match_flash_attention_pattern(node: fx.Node):
    query = key_trans = value = None
    scale_params = (1.0, False)
    attention_mask = None

    is_valid_bmm, attention_weights, value = check_bmm_op(node)
    if not is_valid_bmm:
        return False, []

    if not check_softmax_op(attention_weights):
        return False, []

    softmax_input = get_input_node(attention_weights, 0)
    current_node = softmax_input
    intermediate_nodes = []

    for _ in range(30):
        if not isinstance(current_node, fx.Node) or not current_node.args:
            break

        intermediate_nodes.append(current_node)
        current_node = current_node.args[0]

        is_valid_bmm, query, key_trans = check_bmm_op(current_node)
        if is_valid_bmm:
            break

    is_valid, add_count, _ = validate_operation_counts(intermediate_nodes)
    if not is_valid or not query:
        return False, []

    if check_add_op(softmax_input):
        attention_mask = softmax_input.args[1]
    elif add_count > 0:
        logger.warning("Flash attention pass: Invalid add operation placement")
        return False, []

    for node in intermediate_nodes:
        is_scale_op, _, params = check_div_or_mul_op(node)
        if is_scale_op:
            scale_params = params
            break

    return True, [query, key_trans, value, scale_params, attention_mask]


class FusedFlashAttention(Pattern):
    def process(self, graph_module: fx.GraphModule):
        return False

        modified = False
        graph_module.add_submodule("flash_attn_base", FlashAttentionReplacement())
        graph_module.add_submodule(
            "flash_attn_transpose", FlashAttentionWithTranspose()
        )

        for node in reversed(graph_module.graph.nodes):
            matched, params = match_flash_attention_pattern(node)
            if not matched:
                continue

            modified = True
            query, key_trans, value, scale_params, attention_mask = params

            with graph_module.graph.inserting_before(node):
                fused = graph_module.graph.call_module(
                    "flash_attn_transpose",
                    args=(query, key_trans, value, scale_params, attention_mask),
                )
            node.replace_all_uses_with(fused)

            if validate_transpose_operation(key_trans):
                with graph_module.graph.inserting_before(node):
                    fused = graph_module.graph.call_module(
                        "flash_attn_base",
                        args=(
                            query,
                            key_trans.args[0],
                            value,
                            scale_params,
                            attention_mask,
                        ),
                    )
                node.replace_all_uses_with(fused)

        graph_module.graph.lint()
        graph_module.recompile()
        return modified
