import torch
import operator
from torch import nn, fx


def _is_valid_node(node: fx.Node) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function"


def get_input_node(node, idx):
    return node.args[idx]


def get_actual_node(node, idx):
    new_node = node.args[idx]
    if check_copy_op(new_node):
        return new_node.args[0]
    return new_node


def check_op(node: fx.Node, target) -> bool:
    return _is_valid_node(node) and node.target == target


def check_rsqrt_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.rsqrt.default)


def check_add_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.add.Tensor)


def check_mean_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.mean.dim)


def check_pow_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.pow.Tensor_Scalar)


def check_mul_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.mul.Tensor)


def check_meta(node: fx.Node) -> bool:
    if node.meta == {}:
        return False
    if "tensor_meta" not in node.meta:
        return False
    return True


def check_meta_2d(node: fx.Node) -> bool:
    if not check_meta(node):
        return False
    if len(node.meta["tensor_meta"].shape) == 2:
        return True
    return False


def check_div_or_mul_op(
    node: fx.Node,
) -> tuple[bool, fx.Node | None, tuple[fx.Node | None, bool] | None]:
    if not _is_valid_node(node):
        return False, None, ()

    if node.target not in [
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
    ]:
        return False, None, ()

    is_div = node.target == torch.ops.aten.div.Tensor
    node0 = get_input_node(node, 0)
    node1 = get_input_node(node, 1)
    return True, node0, (node1, is_div)


def check_bmm_op(node: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_op(node, torch.ops.aten.bmm.default):
        return False, None, None

    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    return True, arg1, arg2


def check_mm_op(node: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_op(node, torch.ops.aten.mm.default):
        return False, None, None

    arg1 = node.args[0]
    arg2 = node.args[1]
    return True, arg1, arg2


def check_view(node):
    if (not check_op(node, torch.ops.aten._unsafe_view.default)) and (
        not check_op(node, torch.ops.aten.view.default)
    ):
        return False
    return True


def check_softmax_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten._softmax.default)


def check_cat_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.cat.default)


def check_slice_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.slice.Tensor)


def check_sum_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.sum.dim_IntList)


def check_stack_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.stack.default)


def check_trans_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.transpose.int)


def check_t_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.t.default)


def check_act_op(
    node: fx.Node,
) -> tuple[bool, fx.Node | None, tuple[fx.Node | None, bool] | None]:
    if not _is_valid_node(node):
        return False, None
    if node.target == torch.ops.aten.silu.default:
        return True, "silu"
    if node.target == torch.ops.aten.gelu.default:
        return True, "gelu"
    if node.target == torch.ops.aten.relu.default:
        return True, "relu"
    return False, None


def check_copy_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten._to_copy.default)


def check_getitem_op(node: fx.node) -> bool:
    return check_op(node, operator.getitem)


def check_layernorm_op(node: fx.node) -> bool:
    return check_op(node, torch.ops.aten.native_layer_norm.default)
