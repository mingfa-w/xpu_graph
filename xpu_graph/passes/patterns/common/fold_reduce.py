import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import get_input_kw_node, get_input_node

reduce_tup = (torch.ops.aten.sum.dim_IntList,)

class FoldReduce(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad, FxStage.forward, FxStage.backward]

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target in reduce_tup]

        for reduce in candidates:
            inp = get_input_node(reduce, 0)
            dims = get_input_kw_node(reduce, "dim")
            if not isinstance(dims, list):
                dims = [dims]
            keep_dim = get_input_kw_node(reduce, "keepdim") or False
            if all([inp.meta['tensor_meta'].shape[dim] == 1 for dim in dims]):
                changed = True
                with gm.graph.inserting_before(reduce):
                    if keep_dim:
                        view = gm.graph.call_function(
                            torch.ops.aten.clone.default,
                            args=(
                                inp,
                            )
                        )
                    else:
                        view = gm.graph.call_function(
                            torch.ops.aten.squeeze.dims,
                            args=(
                                inp,
                                dims,
                            )
                        )
                    reduce.replace_all_uses_with(view)
                    gm.graph.erase_node(reduce)

        gm.graph.lint()
        gm.recompile()
        return changed
