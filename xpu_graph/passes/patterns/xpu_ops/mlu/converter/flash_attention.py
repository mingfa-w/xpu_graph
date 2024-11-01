import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from converter_utils import get_input_node

class SFDPReplacement1(nn.Module):
    def forward(self, query, key, value, inv_param, attn_mask_input):
        batch, head_num, seq_len, head_size = query.shape
        seq_len_k = key.shape[-2]
        query = query.reshape(-1, seq_len, head_size).transpose(0, 1)
        key = key.reshape(-1, seq_len_k, head_size).transpose(0, 1)
        value = value.reshape(-1, seq_len_k, head_size).transpose(0, 1)

        inv_scale, is_div = inv_param
        if is_div:
            softmax_scale = 1.0 / inv_scale
        else:
            softmax_scale = inv_scale

        if attn_mask_input is not None:
            attn_mask_input = torch.broadcast_to(
                attn_mask_input, (batch, head_num, seq_len, seq_len_k)
            ).contiguous()
        output = torch_mlu_ops.flash_attention(
            query,
            key,
            value,
            None,
            torch.tensor([0, seq_len], dtype=torch.int32, device="mlu"),
            torch.tensor([0, seq_len_k], dtype=torch.int32, device="mlu"),
            None,
            attn_mask_input,
            seq_len,
            seq_len_k,
            softmax_scale,
            False,
        )

        return (
            output.reshape(-1, seq_len, head_num, head_size)
            .transpose(1, 2)
            .reshape(query.shape)
        )


class SFDPReplacement2(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_model = SFDPReplacement1()

    def forward(self, query, key_trans, value, inv_param, attn_mask_input):
        key = key_trans.transpose(-1, -2)
        return self.inner_model(query, key, value, inv_param, attn_mask_input)

def check_trans_op(key_transpose):
    if not isinstance(key_transpose, fx.Node) or key_transpose.op != "call_function":
        return False

    if key_transpose.target != torch.ops.aten.transpose.int:
        return False

    key, dim1, dim2 = key_transpose.args
    valid_dims = [(-2, -1), (-1, -2), (1, 2), (2, 3)]

    return (dim1, dim2) in valid_dims


def check_div_or_mul_op(softmax_input):
    # Check if node is a div/mul operation
    if not isinstance(softmax_input, fx.Node) or softmax_input.op != "call_function":
        return False, None, ()

    if softmax_input.target not in [
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
    ]:
        return False, None, ()

    is_div = True
    if softmax_input.target != torch.ops.aten.div.Tensor:
        is_div = False
    node0 = get_input_node(softmax_input, 0)
    node1 = get_input_node(softmax_input, 1)
    return True, node0, (node1, is_div)


def check_bmm_op(softmax_input):
    if not isinstance(softmax_input, fx.Node) or softmax_input.op != "call_function":
        return False, None, None

    if softmax_input.target != torch.ops.aten.bmm.default:
        return False, None, None

    query = get_input_node(softmax_input, 0)
    key_transpose = get_input_node(softmax_input, 1)
    return True, query, key_transpose


def check_softmax_op(node):
    if not isinstance(node, fx.Node) or node.op != "call_function":
        return False

    if node.target != torch.ops.aten._softmax.default:
        return False
    return True


def check_add_op(node):
    if not isinstance(node, fx.Node) or node.op != "call_function":
        return False

    if node.target != torch.ops.aten.add.Tensor:
        return False
    return True


def check_used_node(node_list):
    add_count = div_count = mul_count = 0
    for node in node_list:
        if node.target == torch.ops.aten.add.Tensor:
            add_count += 1
        elif node.target == torch.ops.aten.div.Tensor:
            div_count += 1
        elif node.target == torch.ops.aten.mul.Tensor:
            mul_count += 1
        else:
            if node.target not in [
                torch.ops.aten.view.default,
                torch.ops.aten._unsafe_view.default,
                torch.ops.aten.expand.default,
                torch.ops.aten.clone.default,
                torch.ops.aten._to_copy.default,
            ]:
                logger.warning(
                    f"fa pass : node.target: {node.target} is not support in mlu-flashattn"
                )
                return False, 0, 0
    if add_count > 1:
        logger.warning(
            "fa pass : add count: {add_count} is not support in mlu-flashattn"
        )
        return False, 0, 0
    if (div_count + mul_count) > 1:
        logger.warning(
            "fa pass : div count: {div_count + mul_count} is not support in mlu-flashattn"
        )
        return False, 0, 0
    return True, add_count, (div_count + mul_count)


def is_sfdp_pattern_2(node: fx.Node):
    """
    torch.matmul(query, key.transpose(-2, -1))
         .div(inv_scale)
         .softmax(dim=-1)
         .matmul(value)
    """
    changed = False
    query = None
    key_trans = None
    value = None
    inv_param = (1.0, False)
    attn_mask_input = None

    is_bmm, attn_weight, value = check_bmm_op(node)
    if is_bmm:
        if check_softmax_op(attn_weight):
            softmax_input = get_input_node(attn_weight, 0)
            next_bmm = softmax_input
            middle_node = []
            max_count = 30
            while max_count:
                if len(next_bmm.args) == 0:
                    break
                if not isinstance(next_bmm, fx.Node):
                    break
                middle_node.append(next_bmm)
                next_bmm = next_bmm.args[0]
                is_bmm, query, key_trans = check_bmm_op(next_bmm)
                if is_bmm:
                    break
                max_count -= 1

            passed, add_count, _ = check_used_node(middle_node)

            if passed and query:
                changed = True
                if check_add_op(softmax_input):
                    attn_mask_input = softmax_input.args[1]
                else:
                    if add_count > 0:
                        logger.warning(
                            "fa pass : add count is 1 but not ahead softmax is not support in mlu-flashattn"
                        )
                        return False, []
                for idx in range(len(middle_node)):
                    is_div_or_mul, tmp_node, tmp_param = check_div_or_mul_op(
                        middle_node[idx]
                    )
                    if is_div_or_mul:
                        inv_param = tmp_param
                        break

    return changed, [query, key_trans, value, inv_param, attn_mask_input]

class FusedFlashAttention(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        gm.add_submodule("tmo_fa1", SFDPReplacement1())
        gm.add_submodule("tmo_fa2", SFDPReplacement2())

        for node in reversed(gm.graph.nodes):
            matched, param = is_sfdp_pattern_2(node)
            if matched:
                changed = True
                query, key_trans, value, inv_param, attn_mask_input = param
                with gm.graph.inserting_before(node):
                    fused_node = gm.graph.call_module(
                        "tmo_fa2",
                        args=(query, key_trans, value, inv_param, attn_mask_input),
                    )
                node.replace_all_uses_with(fused_node)
                key_trans = fused_node.args[1]
                if check_trans_op(key_trans):
                    with gm.graph.inserting_before(node):
                        fused_node = gm.graph.call_module(
                            "tmo_fa1",
                            args=(
                                query,
                                key_trans.args[0],
                                value,
                                inv_param,
                                attn_mask_input,
                            ),
                        )
                    node.replace_all_uses_with(fused_node)
        gm.graph.lint()
        gm.recompile()
        return changed