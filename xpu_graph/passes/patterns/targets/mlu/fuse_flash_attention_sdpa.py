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
    # check_div_or_mul_op,
    # check_sub_or_add_op,
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


def get_attr(query, key, value, attention_mask):
    if len(query.shape) == 4:
        batch, head_num_q, seq_q, head_size_qk = query.shape
    else:
        head_num_q, seq_q, head_size_qk = query.shape
        if len(key.shape) == 4:
            batch = key.shape[0]
            head_num_q = head_num_q // batch
        elif len(value.shape) == 4:
            batch = value.shape[0]
            head_num_q = head_num_q // batch
        else:
            batch = 1
    seq_kv = key.shape[-2]

    if (
        attention_mask != None
        and len(attention_mask.shape) == 4
        and len(query.shape) == 3
    ):
        batch = attention_mask.shape[0]
        head_num_q = query.shape[0] // batch
    return batch, head_num_q, seq_q, head_size_qk, seq_kv


@torch.library.custom_op("torch_mlu::sdpa_forward", mutates_args=())
def sdpa_forward(
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

    if query.dtype != output_dtype:
        query = query.to(output_dtype)
    if key.dtype != output_dtype:
        key = key.to(output_dtype)
    if value.dtype != output_dtype:
        value = value.to(output_dtype)

    batch, head_num_q, seq_q, head_size_qk, seq_kv = get_attr(
        query, key, value, attention_mask
    )

    query = query.view(batch, head_num_q, seq_q, head_size_qk)
    key = key.view(batch, -1, seq_kv, head_size_qk)
    value = value.view(batch, key.shape[1], seq_kv, -1)

    if attention_mask != None:
        if attention_mask.dtype != output_dtype:
            attention_mask = attention_mask.to(output_dtype)
        if is_add == False:
            attention_mask = torch.neg(attention_mask)
        attention_mask = torch.broadcast_to(
            attention_mask, (batch, head_num_q, seq_q, seq_kv)
        ).contiguous()

    scale_factor = scale_factor.item()
    softmax_scale = 1.0 / scale_factor if is_division else scale_factor

    query = query.view(batch, head_num_q, seq_q, head_size_qk)
    key = key.view(batch, -1, seq_kv, head_size_qk)
    value = value.view(batch, key.shape[1], seq_kv, -1)
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=softmax_scale
    )
    """
    qk = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
    if attention_mask != None:
        if len(attention_mask.shape) == 4:
            qk = qk.view(batch, head_num_q, seq_q, seq_kv)
        qk += attention_mask
    qk = qk.softmax(dim=-1)
    qk = qk.view(-1, seq_q, seq_kv)
    output = torch.bmm(qk, value)
    """
    if output.dtype != output_dtype:
        output = output.to(output_dtype)
    return output.reshape(output_shape)


@sdpa_forward.register_fake
def sdpa_forward_fake(
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


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query,
        key,
        need_key_trans,
        value,
        scale_params,
        add_params,
        output_shape,
        output_dtype,
    ):
        if need_key_trans:
            key = key.transpose(-1, -2)
        scale_factor, is_division = scale_params
        attention_mask, is_add = add_params

        if isinstance(scale_factor, float):
            scale_factor = torch.tensor(scale_factor)

        return sdpa_forward(
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


def _is_sdpa(node: fx.Node):
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

    key = bmm_1_node.args[2]
    need_key_trans = True
    if validate_transpose_operation(get_actual_node(bmm_1_node, 2)):
        key = get_actual_node(bmm_1_node, 2).args[0]
        need_key_trans = False
    return True, [
        bmm_1_node.args[0],
        key,
        need_key_trans,
        node.args[2],
        scale_params,
        add_params,
        get_shape(node),
        get_dtype(node),
    ]


class FusedFlashAttention1(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule):
        graph_module.add_submodule(
            "scaled_dot_product_attention", ScaledDotProductAttention()
        )
        modified = False
        for node in reversed(graph_module.graph.nodes):
            matched, fa_param = _is_sdpa(node)
            if not matched:
                continue
            with graph_module.graph.inserting_before(node):
                fused = graph_module.graph.call_module(
                    "scaled_dot_product_attention",
                    args=tuple(fa_param),
                )

            node.replace_all_uses_with(fused)
            modified = True
        if modified:
            print(graph_module.graph)

        return modified
