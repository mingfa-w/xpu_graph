import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern

class DynamicQuantize(Pattern):

    def process(self, gm: fx.GraphModule):
        from torch.fx import subgraph_rewriter

        def _search_pattern(x):
            return torch.ops.npu.npu_dynamic_quant.default(x)

        def _replacement(x):
            return torch.ops.xpu_ops.dynamic_quantize.default(x)

        match = subgraph_rewriter.replace_pattern(gm, _search_pattern, _replacement)

        return len(match) != 0
