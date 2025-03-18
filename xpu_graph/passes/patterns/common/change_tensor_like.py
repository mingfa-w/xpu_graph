import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern

class ChangeTensorLike(Pattern):
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        changed = False
        tensor_like_map = {
            torch.ops.aten.ones_like.default: torch.ops.aten.ones.default,
            torch.ops.aten.zeros_like.default: torch.ops.aten.zeros.default,
        }
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target in tensor_like_map]

        for like in candidates:
            changed = True
            inp = like.args[0]
            with gm.graph.inserting_before(like):
                tensor = gm.graph.call_function(
                    tensor_like_map[like.target],
                    args=(
                        list(like.meta['tensor_meta'].shape),
                    ),
                    kwargs={
                        'dtype': like.meta['tensor_meta'].dtype,
                        'device': like.meta['val'].device,
                        'pin_memory': like.kwargs['pin_memory'],
                    },
                )
            like.replace_all_uses_with(tensor)
            gm.graph.erase_node(like)

        gm.graph.lint()
        gm.recompile()
        return changed
