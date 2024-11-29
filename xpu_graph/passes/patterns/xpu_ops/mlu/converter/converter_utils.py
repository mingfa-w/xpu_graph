import torch
from torch import nn, fx


def _is_valid_node(node: fx.Node) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function"


def get_input_node(node, idx):
    return node.args[idx]


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


def check_div_or_mul_op(
    softmax_input: fx.Node,
) -> tuple[bool, fx.Node | None, tuple[fx.Node | None, bool] | None]:
    if not _is_valid_node(softmax_input):
        return False, None, ()

    if softmax_input.target not in [
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
    ]:
        return False, None, ()

    is_div = softmax_input.target == torch.ops.aten.div.Tensor
    node0 = get_input_node(softmax_input, 0)
    node1 = get_input_node(softmax_input, 1)
    return True, node0, (node1, is_div)


def check_bmm_op(softmax_input: fx.Node) -> tuple[bool, fx.Node | None, fx.Node | None]:
    if not check_op(softmax_input, torch.ops.aten.bmm.default):
        return False, None, None

    query = get_input_node(softmax_input, 0)
    key_transpose = get_input_node(softmax_input, 1)
    return True, query, key_transpose


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
