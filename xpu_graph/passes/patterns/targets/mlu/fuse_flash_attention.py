import torch
from torch import nn, fx
import torch_mlu
from typing import List, Tuple

from xpu_graph import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
import torch.nn.functional as F
from ...utils.check_ops import (
    check_copy,
    check_add_op,
    check_bmm_op,
    check_softmax_op,
    get_actual_node,
    get_shape,
    get_dtype,
    check_view,
    _is_valid_node,
    get_input_node,
)


def check_div_or_mul_op(
    node: fx.Node,
):
    if not _is_valid_node(node):
        return False, None

    if node.target not in [
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
    ]:
        return False, None

    is_div = node.target == torch.ops.aten.div.Tensor
    return True, is_div


def check_sub_or_add_op(
    node: fx.Node,
):
    if not _is_valid_node(node):
        return False, None

    if node.target not in [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
    ]:
        return False, None

    is_add = node.target == torch.ops.aten.add.Tensor
    return True, is_add


@torch.library.custom_op("torch_mlu::tmo_fa_forward", mutates_args=())
def tmo_fa_forward1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scale_factor: torch.Tensor,
    is_division: bool,
    is_add: bool,
    output_shape: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    import torch_mlu_ops

    if query.dtype != output_dtype:
        query = query.to(output_dtype)
    if key.dtype != output_dtype:
        key = key.to(output_dtype)
    if value.dtype != output_dtype:
        value = value.to(output_dtype)

    if len(query.shape) == 4:
        batch_size, num_heads, sequence_len, head_dim = query.shape
    else:
        num_heads, sequence_len, head_dim = query.shape
        batch_size = 1
    key_sequence_len = key.shape[-2]

    if attention_mask != None:
        if attention_mask.dtype != output_dtype:
            attention_mask = attention_mask.to(output_dtype)
        if is_add == False:
            attention_mask = torch.neg(attention_mask)
        if len(attention_mask.shape) == 4:
            if len(query.shape) == 3:
                batch_size = attention_mask.shape[0]
                num_heads = query.shape[0] // batch_size
            attention_mask = torch.broadcast_to(
                attention_mask, (batch_size, -1, sequence_len, key_sequence_len)
            ).contiguous()

    scale_factor = scale_factor.item()
    softmax_scale = 1.0 / scale_factor if is_division else scale_factor

    if 0:  # num_heads <= 128:
        if len(query.shape) == 4:
            query = query.transpose(2, 1)
            key = key.transpose(2, 1)
            value = value.transpose(2, 1)
        else:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        key = key.contiguous()
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
        output = output.reshape(-1, sequence_len, num_heads, head_dim).transpose(1, 2)
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
        return output.view(output_shape)
    else:
        qk = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
        if attention_mask != None:
            if len(attention_mask.shape) == 4:
                qk = qk.view(batch_size, num_heads, sequence_len, key_sequence_len)
            qk += attention_mask
        qk = qk.softmax(dim=-1)
        qk = qk.view(-1, sequence_len, key_sequence_len)
        output = torch.bmm(qk, value)
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
        return output.view(output_shape)


@torch.library.custom_op("torch_mlu::tmo_fa_forward", mutates_args=())
def tmo_fa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scale_factor: torch.Tensor,
    is_division: bool,
    is_add: bool,
    output_shape: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    import torch_mlu_ops

    if query.dtype != output_dtype:
        query = query.to(output_dtype)
    if key.dtype != output_dtype:
        key = key.to(output_dtype)
    if value.dtype != output_dtype:
        value = value.to(output_dtype)

    num_heads, sequence_len, head_dim = query.shape
    batch_size = 1
    key_sequence_len = key.shape[-2]
    if attention_mask != None and len(attention_mask.shape) == 4:
        if len(query.shape) == 3:
            batch_size = attention_mask.shape[0]
            num_heads = query.shape[0] // batch_size

    if attention_mask != None:
        if attention_mask.dtype != output_dtype:
            attention_mask = attention_mask.to(output_dtype)
        if is_add == False:
            attention_mask = torch.neg(attention_mask)
        if len(attention_mask.shape) == 4:
            attention_mask = torch.broadcast_to(
                attention_mask, (batch_size, -1, sequence_len, key_sequence_len)
            ).contiguous()

    scale_factor = scale_factor.item()
    softmax_scale = 1.0 / scale_factor if is_division else scale_factor

    query = query.view(batch_size, num_heads, sequence_len, head_dim)
    key = key.view(batch_size, -1, key_sequence_len, head_dim)
    value = value.view(batch_size, key.shape[1], key_sequence_len, -1)
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=softmax_scale
    )
    """
    qk = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
    if attention_mask != None:
        if len(attention_mask.shape) == 4:
            qk = qk.view(batch_size, num_heads, sequence_len, key_sequence_len)
        qk += attention_mask
    qk = qk.softmax(dim=-1)
    qk = qk.view(-1, sequence_len, key_sequence_len)
    output = torch.bmm(qk, value)
    """
    if output.dtype != output_dtype:
        output = output.to(output_dtype)
    return output.view(output_shape)


@tmo_fa_forward.register_fake
def tmo_fa_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scale_factor: torch.Tensor,
    is_division: bool,
    is_add: bool,
    output_shape: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    output_tensor = torch.empty(
        output_shape,
        dtype=output_dtype,
        device=query.device,
    )
    return output_tensor


class FlashAttention(nn.Module):
    def forward(
        self,
        query,
        key_transposed,
        value,
        scale_params,
        add_params,
        output_shape,
        output_dtype,
    ):
        key = key_transposed.transpose(-1, -2)
        scale_factor, is_division = scale_params
        attention_mask, is_add = add_params

        if isinstance(scale_factor, float):
            scale_factor = torch.tensor(scale_factor)

        return tmo_fa_forward(
            query,
            key,
            value,
            attention_mask,
            scale_factor,
            is_division,
            is_add,
            output_shape,
            output_dtype,
        )


def validate_transpose_operation(key_transpose):
    if not isinstance(key_transpose, fx.Node) or key_transpose.op != "call_function":
        return False

    if key_transpose.target != torch.ops.aten.transpose.int:
        return False

    dim1, dim2 = key_transpose.args[1:]
    valid_dimensions = [(-2, -1), (-1, -2), (1, 2), (2, 3)]
    return (dim1, dim2) in valid_dimensions


def _is_fa(node: fx.Node):
    if node.target != "mlu_tmo_fused_bmm_replacement":
        return False, []
    softmax_node = get_actual_node(node, 0)
    if not check_softmax_op(softmax_node):
        return False, []

    bmm_1_node = get_actual_node(softmax_node, 0)

    # (optional) find add
    add_params = (None, False)
    is_scale_op, is_add = check_sub_or_add_op(bmm_1_node)
    if is_scale_op:
        add_params = (get_input_node(bmm_1_node, 1), is_add)
        bmm_1_node = get_actual_node(bmm_1_node, 0)

    # (optional) find div or mul
    scale_params = (1.0, False)
    is_scale_op, is_div = check_div_or_mul_op(bmm_1_node)
    if is_scale_op:
        scale_params = (get_input_node(bmm_1_node, 1), is_div)
        bmm_1_node = get_actual_node(bmm_1_node, 0)

    if bmm_1_node.target != "mlu_tmo_fused_bmm_replacement":
        if bmm_1_node.target != "mlu_tmo_fused_bmm_add_replacement":
            return False, []
        if add_params[0] != None:
            logger.warning("Flash attention pass: Too many add operations")
            return False, []
        add_params[0] = bmm_1_node.args[2]
        add_params[1] = True

    return True, [
        bmm_1_node.args[0],
        bmm_1_node.args[2],
        node.args[2],
        scale_params,
        add_params,
        get_shape(node),
        get_dtype(node),
    ]


class FusedFlashAttention(Pattern):
    _opt_level = OptLevel.level2
    """
    bmm+softmax+bmm->fa
    """

    def process(self, graph_module: fx.GraphModule):
        graph_module.add_submodule("flash_attn", FlashAttention())
        modified = False
        for node in reversed(graph_module.graph.nodes):
            matched, fa_param = _is_fa(node)
            if not matched:
                continue
            with graph_module.graph.inserting_before(node):
                fused = graph_module.graph.call_module(
                    "flash_attn",
                    args=tuple(fa_param),
                )

            node.replace_all_uses_with(fused)
            modified = True

        return modified
