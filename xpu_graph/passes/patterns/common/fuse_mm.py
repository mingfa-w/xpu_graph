from typing import Optional, Union, Tuple

import torch
from torch import nn, fx
import torch.nn.functional as F
from typing import Callable, Optional, List
from xpu_graph.config import OptLevel

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from xpu_graph.fx_utils import FxStage

from ..utils.check_ops import (
    check_add_op,
    check_mm_op,
)


class FusedAddMM(Pattern):
    _opt_level = OptLevel.level2
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        for node in reversed(graph_module.graph.nodes):
            is_add = check_add_op(node)
            if not is_add:
                continue
            mm_node = node.args[0]
            bias_node = node.args[1]
            # Note: This pattern does not fuse residuals
            is_mm, input_node, weight_node = check_mm_op(mm_node)
            if not is_mm:
                mm_node = node.args[1]
                bias_node = node.args[0]
                is_mm, input_node, weight_node = check_mm_op(mm_node)
                if not is_mm:
                    continue
            if len(mm_node.users) != 1:
                continue

            if isinstance(bias_node, float) or isinstance(bias_node, int):
                continue

            with graph_module.graph.inserting_before(node):
                addmm_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.addmm.default,
                    args=(bias_node, input_node, weight_node),
                    name=node.name + "_replacement",
                )

            node.replace_all_uses_with(addmm_node)
            changed = True

        graph_module.graph.lint()
        graph_module.recompile()
        return changed
