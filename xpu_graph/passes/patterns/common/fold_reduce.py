import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern

reduce_tup = (torch.ops.aten.sum.dim_IntList,)

class FoldReduce(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target in reduce_tup]

        for reduce in candidates:
            inp = reduce.args[0]
            # dims may be List[Int] or None.
            dims = reduce.args[1]
            if dims is None:
                continue
            dim = dims[0]
            if inp.meta['tensor_meta'].shape[dim] == 1:
                changed = True
                with gm.graph.inserting_before(reduce):
                    view = gm.graph.call_function(
                        torch.ops.aten.view.default,
                        args=(
                            inp,
                            reduce.meta['tensor_meta'].shape
                        )
                    )
                    reduce.replace_all_uses_with(view)
                    gm.graph.erase_node(reduce)

        gm.graph.lint()
        gm.recompile()
        return changed