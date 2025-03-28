import torch
import torch.fx as fx

def _is_full_like_node_v(node: fx.Node, value: int):
    if node.op != "call_function":
        return False
    is_full, value_ = _is_full_like_node(node)
    if is_full:
        return value_ == value
    return False 

def _is_full_like_node(node: fx.Node):
    if node.op != "call_function":
        return False, None

    if (
        node.target == torch.ops.aten.full.default
        or node.target == torch.ops.aten.full_like.default
    ):
        if len(node.args) >= 2:
            full_value = node.args[1]
            return True, full_value 

    if node.target == torch.ops.aten.ones_like.default:
        return True, 1
    if node.target == torch.ops.aten.zeros_like.default:
        return True, 0

    if node.target == torch.ops.aten.ones.default:
        return True, 1
    if node.target == torch.ops.aten.zeros.default:
        return True, 0

    return False, None
