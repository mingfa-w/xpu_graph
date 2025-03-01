import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern_macher import Matcher, PatMatcher

class GeluNew(PatMatcher):
    def rewriter(self, gm: fx.GraphModule) -> bool:
        # Step 1: create a matcher with pattern
        matcher = Matcher(gm,
            """
                x = placeholder(); x = placeholder
                div = torch.ops.aten.div.Tensor(x, 1.4142135623730951)
                erf = torch.ops.aten.erf.default(div);  div = None
                add = torch.ops.aten.add.Tensor(erf, 1);  erf = None
                mul = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
                ? = torch.ops.aten.mul.Tensor(mul, x);  mul = arg0_1 = None
            """
        )
        matcher.print_matchs()
        
        # Step 2: run replace
        def replacement(matcher, isubgraph, subgraph):
            print("GeluNew replacement")       
        matcher.replace(replacement)
        
        return False

