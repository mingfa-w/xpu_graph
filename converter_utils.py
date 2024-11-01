import torch
from torch import nn, fx


def skip_view(node: fx.Node):
    if not isinstance(node, fx.Node):
        return node

    skip_targets = {
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.expand.default,
    }

    clone_targets = {torch.ops.aten.clone.default, torch.ops.aten._to_copy.default}

    inplace_targets = {
        torch.ops.aten.add_,
        torch.ops.aten.sub_,
        torch.ops.aten.mul_,
        torch.ops.aten.div_,
    }

    while isinstance(node, fx.Node):
        if node.op != "call_function":
            break
        if node.target in skip_targets:
            node = node.args[0]
            continue
        if node.target in clone_targets:
            # Check if any user is not an inplace op
            if any(
                user.target not in inplace_targets
                for user in node.users
                if user.op == "call_function"
            ):
                node = node.args[0]
                continue
        break
    return node

def get_input_node(node, idx):
    return skip_view(node.args[idx])
