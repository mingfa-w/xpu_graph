from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch import SymBool, SymInt, SymFloat
import torch.fx as fx

from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.config import OptLevel
from xpu_graph.utils import logger
from xpu_graph.constant_manager import get_constant_manager
from xpu_graph.fx_utils import get_disable_fake_mode_handler

__all__ = ["ConstantFolding"]


def _no_folding(node: fx.Node):
    no_fold_call_function_list = [
        torch.ops.aten.t.default,
        torch.ops.aten.lift_fresh_copy.default,
    ]
    if node.op == "call_function":
        return node.target in no_fold_call_function_list


class ConstantFolding(Optimizer):

    def __init__(self, config):
        super().__init__()
        self._is_training = config.is_training

    def _is_constant(self, arg: Any, gm: fx.GraphModule):
        if isinstance(arg, fx.Node):
            # Note: For safety, bypass all parameters when is training.
            if self._is_training:
                if arg.op != "get_attr":
                    return False
                assert hasattr(gm, arg.target)
                return not isinstance(getattr(gm, arg.target), torch.nn.Parameter)
            return arg.op == "get_attr"
        if type(arg) in (SymBool, SymInt, SymFloat):
            return False
        return True

    def _all_inputs_constant(self, node: fx.Node, gm: fx.GraphModule):
        flattern_args, _ = tree_flatten(node.args)
        flattern_kargs, _ = tree_flatten(node.kwargs)

        return all(self._is_constant(arg, gm) for arg in flattern_args + flattern_kargs)

    def _get_real_inputs(self, node: fx.Node, gm: fx.GraphModule):
        flattern_args, args_spec = tree_flatten(node.args)
        flattern_kwargs, kwargs_spec = tree_flatten(node.kwargs)

        new_flattern_args = flattern_args[:]
        new_flattern_kwargs = flattern_kwargs[:]

        for args, new_args in zip(
            [flattern_args, flattern_kwargs], [new_flattern_args, new_flattern_kwargs]
        ):
            for i, arg in enumerate(args):
                if isinstance(arg, fx.Node):
                    assert arg.op == "get_attr"
                    assert hasattr(gm, arg.target)
                    new_args[i] = getattr(gm, arg.target)

        return tree_unflatten(new_flattern_args, args_spec), tree_unflatten(
            new_flattern_kwargs, kwargs_spec
        )

    def process(self, gm: torch.fx.GraphModule):
        changed = False
        graph = gm.graph
        get_attr_insert_point = None

        # For better readability, we insert get_attr node in the front of graph
        for get_attr_insert_point in gm.graph.nodes:
            if (
                get_attr_insert_point.op != "get_attr"
                and get_attr_insert_point.op != "placeholder"
            ):
                break

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if _no_folding(node):
                continue
            if self._all_inputs_constant(node, gm):
                changed = True

                logger.info(f"start constant folding: {node.name} {node.target}")

                real_args, real_kwargs = self._get_real_inputs(node, gm)

                disable_fake_mode = get_disable_fake_mode_handler()
                with disable_fake_mode():
                    constant_value = node.target(*real_args, **real_kwargs)

                constant_name = node.name + "_constant_folding"
                constant_name = get_constant_manager(gm).register_constant(
                    constant_value, constant_name
                )
                with graph.inserting_before(get_attr_insert_point):
                    constant_node = graph.create_node("get_attr", constant_name)
                    node.replace_all_uses_with(constant_node)

        changed = get_constant_manager(gm).remove_redundant_constants() or changed

        gm.graph.lint()

        return changed
