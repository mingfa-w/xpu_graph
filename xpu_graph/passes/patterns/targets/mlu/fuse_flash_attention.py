import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops
from typing import List, Tuple

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    check_copy,
    check_view,
    check_clone,
    check_add_op,
    check_bmm_op,
    check_softmax_op,
    get_actual_node,
    get_shape,
    get_dtype,
    check_div_or_mul_op,
    check_sub_or_add_op,
)


def calc_params(scale_params, params):
    params = list(params)
    if scale_params[0] == 1.0:
        return params
    if scale_params[1] == params[1]:
        return [scale_params[0] * params[0], scale_params[1]]
    else:
        return [scale_params[0] / params[0], scale_params[1]]


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
    need_postprocess: bool,
) -> torch.Tensor:
    batch_size, sequence_len, num_heads, head_dim = query.shape
    key_sequence_len = key.shape[-3]

    if isinstance(scale_factor, torch._subclasses.FakeTensor):
        scale_factor = 1.0
    else:
        scale_factor = scale_factor.item()
    softmax_scale = 1.0 / scale_factor if is_division else scale_factor

    if num_heads <= 128:
        # key = key.contiguous()
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
        if need_postprocess:
            output = output.transpose(1, 2)
    else:
        query = query.transpose(2, 1)
        key = key.transpose(2, 1)
        value = value.transpose(2, 1)
        query = query.view(-1, sequence_len, head_dim)
        key = key.view(-1, key_sequence_len, head_dim)
        value = value.view(-1, key_sequence_len, head_dim)
        qk = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
        if attention_mask != None:
            attention_mask = attention_mask.view(-1, sequence_len, key_sequence_len)
            qk += attention_mask
        qk = qk.softmax(dim=-1)
        qk = qk.view(-1, sequence_len, key_sequence_len)
        output = torch.bmm(qk, value)
        if not need_postprocess:
            output = output.reshape(
                batch_size, -1, output.shape[-2], output.shape[-1]
            ).transpose(1, 2)
    return output


class FlashAttentionReplacement(nn.Module):
    def forward(
        self,
        query,
        key,
        value,
        scale_params,
        add_params,
        output_shape,
        output_dtype,
        need_key_trans,
        need_value_trans,
        need_postprocess,
    ):
        scale_factor, is_division = scale_params
        attention_mask, is_add = add_params

        if isinstance(scale_factor, float):
            scale_factor = torch.tensor(scale_factor)
        if need_key_trans:
            key = key.transpose(-1, -2)

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
            query = query.unsqueeze(0)
        key_sequence_len = key.shape[-2]
        if len(key.shape) == 3:
            key = key.unsqueeze(0)
        if len(value.shape) == 3:
            value = value.unsqueeze(0)

        query = query.transpose(2, 1)
        key = key.transpose(2, 1)
        if need_value_trans:
            value = value.transpose(2, 1)

        if attention_mask != None:
            if attention_mask.dtype != output_dtype:
                attention_mask = attention_mask.to(output_dtype)
            if is_add == False:
                attention_mask = torch.neg(attention_mask)
            attention_mask = torch.broadcast_to(
                attention_mask, (batch_size, num_heads, sequence_len, key_sequence_len)
            ).contiguous()

        output = tmo_fa_forward(
            query,
            key,
            value,
            attention_mask,
            scale_factor,
            is_division,
            is_add,
            output_shape,
            output_dtype,
            need_postprocess,
        )
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
        if output.shape != output_shape:
            output = output.reshape(output_shape)
        return output


def validate_output_transpose_operation(output_transpose):
    if len(get_shape(output_transpose)) != 4:
        return False
    if (
        not isinstance(output_transpose, fx.Node)
        or output_transpose.op != "call_function"
    ):
        return False

    if output_transpose.target != torch.ops.aten.transpose.int:
        return False

    dim1, dim2 = output_transpose.args[1:]
    valid_dimensions = [(-2, -3), (1, 2)]
    return (dim1, dim2) in valid_dimensions


def validate_key_transpose_operation(key_transpose):
    if not isinstance(key_transpose, fx.Node) or key_transpose.op != "call_function":
        return False

    if key_transpose.target != torch.ops.aten.transpose.int:
        return False

    dim1, dim2 = key_transpose.args[1:]
    if len(get_shape(key_transpose)) == 4:
        valid_dimensions = [(-2, -1), (-1, -2), (2, 3), (3, 2)]
        return (dim1, dim2) in valid_dimensions
    elif len(get_shape(key_transpose)) == 3:
        valid_dimensions = [(-2, -1), (-1, -2), (1, 2), (2, 1)]
        return (dim1, dim2) in valid_dimensions
    else:
        return False


def validate_value_transpose_operation(value):
    if not isinstance(value, fx.Node) or value.op != "call_function":
        return False

    if value.target != torch.ops.aten.transpose.int:
        return False

    if len(get_shape(value)) != 4:
        return False

    dim1, dim2 = value.args[1:]
    valid_dimensions = [(1, 2), (-2, -3), (-3, -2), (2, 1)]
    return (dim1, dim2) in valid_dimensions


def _is_fa(node: fx.Node):
    if node.target != "fused_bmm":
        return False, []
    softmax_node = get_actual_node(node, 0)
    if not check_softmax_op(softmax_node):
        return False, []

    bmm_1_node = get_actual_node(softmax_node, 0)

    # (optional) find add
    add_params = (None, False)
    is_scale_op, addinput1, params = check_sub_or_add_op(bmm_1_node)
    if is_scale_op:
        add_params = params
        bmm_1_node = addinput1

    # (optional) find div or mul
    scale_params = (1.0, False)
    is_scale_op, div_input_node, params = check_div_or_mul_op(bmm_1_node)
    if is_scale_op:
        scale_params = params
        bmm_1_node = div_input_node

    if bmm_1_node.target != "fused_bmm":
        if bmm_1_node.target != "fused_baddbmm":
            return False, []
        if add_params[0] != None:
            logger.warning("Flash attention pass: Too many add operations")
            return False, []
        add_params[0] = bmm_1_node.args[2]
        add_params[1] = True

    changed1 = True
    query = bmm_1_node.args[0]
    key = bmm_1_node.args[1]
    value = node.args[1]
    need_value_trans = True
    need_key_trans = True
    while changed1:
        changed1 = False
        # (optional) find div or mul before query and key
        is_scale_op, div_input_node_, params = check_div_or_mul_op(query)
        if is_scale_op and isinstance(params[0], float):
            scale_params = calc_params(scale_params, params)
            query = query.args[0]
            changed1 = True
        is_scale_op, div_input_node_, params = check_div_or_mul_op(key)
        if is_scale_op and isinstance(params[0], float):
            scale_params = calc_params(scale_params, params)
            key = key.args[0]
            changed1 = True

        # (optional) skip copy
        if check_copy(query):
            query = query.args[0]
            changed1 = True
        if check_copy(key):
            key = key.args[0]
            changed1 = True

        # (optional) skip key_trans
        if validate_key_transpose_operation(key):
            need_key_trans = False
            key = key.args[0]
            changed1 = True

        # (optional) skip value_trans
        if validate_value_transpose_operation(value):
            need_value_trans = False
            value = value.args[0]
            changed1 = True

    return True, [
        query,
        key,
        value,
        scale_params,
        add_params,
        list(node.args[4]),  # shape
        node.args[5],  # dtype
        need_key_trans,
        need_value_trans,
        True,  # need output post process
    ]


# (optional) output_trans+clone+view 4d->3d
# %transpose : [num_users=1] = call_function[target=torch.ops.aten.transpose.int](args = (%flash_attn_base, 1, 2), kwargs = {})
# %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%transpose,), kwargs = {memory_format: torch.contiguous_format})
# %_unsafe_view : [num_users=1] = call_function[target=torch.ops.aten._unsafe_view.default](args = (%clone), kwargs = {})
def _has_postprocess(node: fx.Node):
    if node.target != "flash_attn_base":
        return False, []
    if len(node.users) != 1:
        return False, []
    changed1 = False
    next_node = next(iter(node.users))
    while 1:
        if len(next_node.users) != 1:
            break
        # (optional) skip copy and view
        if (
            check_copy(next_node)
            or check_view(next_node)
            or check_clone(next_node)
            or validate_output_transpose_operation(next_node)
        ):
            changed1 = True
            last_node = next_node
            next_node = next(iter(next_node.users))
        else:
            break
    if changed1:
        return True, [last_node, get_shape(last_node), get_dtype(last_node)]
    else:
        return False, []


def insert_fa_node(graph_module, node, fa_param):
    with graph_module.graph.inserting_before(node):
        fused = graph_module.graph.call_module(
            "flash_attn_base",
            args=tuple(fa_param),
        )

    node.replace_all_uses_with(fused)
    graph_module.graph.erase_node(node)


class FusedFlashAttention(Pattern):
    def process(self, graph_module: fx.GraphModule):
        graph_module.add_submodule("flash_attn_base", FlashAttentionReplacement())
        modified = False
        for node in reversed(graph_module.graph.nodes):
            matched, fa_param = _is_fa(node)
            if not matched:
                continue
            insert_fa_node(graph_module, node, fa_param)
            modified = True

        for node in reversed(graph_module.graph.nodes):
            matched, param = _has_postprocess(node)
            if not matched:
                continue
            fa_param = list(node.args)
            last_node, new_output_shape, new_output_dtype = param
            fa_param[-1] = False
            fa_param[5] = new_output_shape
            fa_param[6] = new_output_dtype
            insert_fa_node(graph_module, last_node, fa_param)

        graph_module.graph.lint()
        graph_module.recompile()
        print(graph_module.graph)
        return modified
