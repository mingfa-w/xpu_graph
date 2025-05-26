import torch
from torch import nn, fx
import torch_mlu
from typing import List, Tuple

from xpu_graph import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from ...utils.submodule_manager import register_new_submodule
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


@torch.library.custom_op("torch_mlu::tmo_fa_forward", mutates_args=())
def tmo_fa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    softmax_scale: float,
    is_add: bool,
    output_shape: List[int],
    output_dtype: torch.dtype,
    need_output_trans: bool,
) -> torch.Tensor:
    import torch_mlu_ops

    if query.dtype != output_dtype:
        query = query.to(output_dtype)
    if key.dtype != output_dtype:
        key = key.to(output_dtype)
    if value.dtype != output_dtype:
        value = value.to(output_dtype)

    batch, seq_q, head_num_q, head_size_qk = query.shape
    seq_kv = key.shape[1]

    if attention_mask != None:
        if attention_mask.dtype != output_dtype:
            attention_mask = attention_mask.to(output_dtype)
        if is_add == False:
            attention_mask = torch.neg(attention_mask)
        attention_mask = torch.broadcast_to(
            attention_mask, (batch, head_num_q, seq_q, seq_kv)
        ).contiguous()

    output = torch_mlu_ops.flash_attention(
        query,
        key,
        value,
        None,
        torch.tensor([0, seq_q], dtype=torch.int32, device="mlu"),
        torch.tensor([0, seq_kv], dtype=torch.int32, device="mlu"),
        None,
        attention_mask,
        seq_q,
        seq_kv,
        softmax_scale,
        False,
    )
    if need_output_trans:
        output = output.reshape(-1, seq_q, head_num_q, head_size_qk).transpose(1, 2)
    if output.dtype != output_dtype:
        output = output.to(output_dtype)
    return output.reshape(output_shape)


@tmo_fa_forward.register_fake
def tmo_fa_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    softmax_scale: float,
    is_add: bool,
    output_shape: List[int],
    output_dtype: torch.dtype,
    need_output_trans: bool,
) -> torch.Tensor:
    output_tensor = torch.empty(
        output_shape,
        dtype=output_dtype,
        device=query.device,
    )
    return output_tensor


class TMOFlashAttention(nn.Module):
    def __init__(self, scale_factor, is_division):
        super().__init__()
        if not isinstance(scale_factor, float):
            scale_factor = scale_factor.item()
        self.softmax_scale = 1.0 / scale_factor if is_division else scale_factor

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
        need_output_trans,
    ):
        if need_key_trans:
            key = key.transpose(-1, -2)
        attention_mask, is_add = add_params

        return tmo_fa_forward(
            query,
            key,
            value,
            attention_mask,
            self.softmax_scale,
            is_add,
            output_shape,
            output_dtype,
            need_output_trans,
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


def merge_scale(scale_param, is_div, factor):
    old_factor, old_is_div = scale_param
    if old_is_div == is_div:
        return (old_factor * factor, is_div)
    else:
        new_factor = old_factor / factor
        new_is_div = old_is_div
        if new_factor < 1:
            new_is_div = not new_is_div
            new_factor = factor / old_factor
        return (new_factor, new_is_div)


def find_actual_output(node):
    while True:
        if len(node.users) == 1:
            if check_view(node):
                node = next(iter(node.users))
            else:
                break
        else:
            break
    return node


def _is_tmofa(node):
    fa_param = list(node.args)
    query = get_actual_node(node, 0)
    key = get_actual_node(node, 1)
    value = get_actual_node(node, 3)

    is_scale_op, is_div = check_div_or_mul_op(query)
    if is_scale_op:
        scale_param = node.args[4]
        new_scale_param = merge_scale(scale_param, is_div, get_input_node(query, 1))
        query = get_actual_node(query, 0)

    for n in (query, key, value):
        if len(get_shape(n)) != 4:
            return False, None

        if n.target == torch.ops.aten.permute.default and n.args[1] == [
            0,
            2,
            1,
            3,
        ]:
            continue
        elif n.target == torch.ops.aten.transpose.int and n.args[1:] in [
            (1, 2),
            (2, 1),
        ]:
            continue
        else:
            return False, None

    # tmo fa limit
    head_num_q = get_shape(query)[2]
    if head_num_q > 128:
        return False, None

    fa_param[0] = query.args[0]
    fa_param[1] = key.args[0]
    fa_param[3] = value.args[0]
    if is_scale_op:
        fa_param[4] = new_scale_param
    return True, fa_param + [True]


class FusedFlashAttention2(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule):
        modified = False
        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_module" and "scaled_dot_product_attention" in node.name
        ]
        for node in candidates:
            matched, fa_param = _is_tmofa(node)
            if not matched:
                continue
            scale_factor, is_div = fa_param[4]
            module_name = register_new_submodule(
                graph_module,
                "tmo_flash_attention",
                TMOFlashAttention,
                args=(scale_factor, is_div),
            )
            with graph_module.graph.inserting_before(node):
                fused = graph_module.graph.call_module(
                    module_name,
                    args=tuple(fa_param),
                )
            node.replace_all_uses_with(fused)
            modified = True

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.transpose.int
        ]
        for node in candidates:
            if len(get_shape(node)) != 4:
                continue
            input_node = get_actual_node(node, 0)
            if len(input_node.users) != 1:
                continue
            if input_node.op != "call_module":
                continue
            if "tmo_flash_attention" in input_node.name:
                fa_param = list(input_node.args)
                fa_param[-3] = get_shape(node)
                fa_param[-2] = get_dtype(node)
                fa_param[-1] = False
                scale_factor, is_div = fa_param[4]
                module_name = register_new_submodule(
                    graph_module,
                    "tmo_flash_attention",
                    TMOFlashAttention,
                    args=(scale_factor, is_div),
                )
                with graph_module.graph.inserting_before(node):
                    fused = graph_module.graph.call_module(
                        module_name,
                        args=tuple(fa_param),
                    )
                node.replace_all_uses_with(fused)
                modified = True

        if modified:
            print(graph_module.graph)
        return modified
