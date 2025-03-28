import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from ..utils.check_ops import (
    check_unsqueeze_op,
    check_squeeze_op,
)

def match(a, b):
    return a == b if isinstance(b, int) else (len(b) == 1 and a in b)

def find_chains(candidates):
    all_chains = []
    visited = set()
    for node in candidates:
        if node in visited:
            continue
        chain = []

        while check_squeeze_op(node) and node not in visited:
            chain.append(node)
            visited.add(node)
            node = node.args[0]
        if len(chain) > 1:
            all_chains.append(chain)
    return all_chains
        
def _is_fold_squeeze(gm, all_chains) -> bool:
    changed = False
    for chain in all_chains:
        keep_node = next((n for n in chain if len(n.args) == 1), None)

        if keep_node:
            ori_input = chain[-1].args[0]
            keep_node.args = (ori_input,)

            for node in chain:
                if node is not keep_node:
                    node.replace_all_uses_with(keep_node)
                    gm.graph.erase_node(node)
            changed = True
    return changed

class FoldSqueeze0(Pattern):
    """
    Fold aten.squeeze(aten.squeeze)
    """

    _stages = [FxStage.inference, FxStage.pregrad, FxStage.forward, FxStage.backward]

    def process(self, gm: fx.GraphModule):
        changed = False

        candidates = [
            node
            for node in reversed(gm.graph.nodes)
            if check_squeeze_op(node)
        ]
        if len(candidates) == 1:
            return False
        all_chains = find_chains(candidates)

        changed = _is_fold_squeeze(gm, all_chains)

        gm.graph.lint()
        gm.recompile()
        return changed

class FoldSqueeze1(Pattern):
    """
    Fold aten.squeeze(aten.unsqueeze)
    """

    _stages = [FxStage.inference, FxStage.pregrad, FxStage.forward, FxStage.backward]

    def process(self, gm: fx.GraphModule):
        changed = False
        for node in reversed(gm.graph.nodes):
            if not check_squeeze_op(node):
                continue
            unsqueeze = node.args[0]
            if not check_unsqueeze_op(unsqueeze):
                continue
            if len(unsqueeze.users) > 1:
                continue

            if len(node.args) == 1:
                changed = True
                node.replace_input_with(unsqueeze, unsqueeze.args[0])
            elif match(unsqueeze.args[1], node.args[1]):
                changed = True
                node.replace_all_uses_with(unsqueeze.args[0])

        gm.graph.lint()
        gm.recompile()
        return changed
