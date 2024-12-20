import typing

import torch
import torch.fx as fx
from torch.utils._pytree import tree_flatten
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import meta_table, pre_autograd_decomposition_table, decomposition_table

from xpu_graph.passes.optimizer import Optimizer

class InlineModuleAndDecomp(fx.Transformer):
    def call_module(self, target, args, kwargs):
        # import pdb;pdb.set_trace()
        mod = self.fetch_attr(target)

        try:
            from torch._dispatch.python import enable_python_dispatcher
            with enable_python_dispatcher():
                traced = make_fx(mod, meta_table, tracing_mode="real")(*args, **kwargs)
        except:
            traced = mod
        # print(f"traced: {traced.graph}")
        return traced(*args, **kwargs)


# class InlineModuleAndDecomp(Optimizer):
#     def process(self, gm: fx.GraphModule):
#         call_module_cnt = 0
#         for n in gm.graph.nodes:
#             if n.op == 'call_module':
#                 call_module_cnt += 1

#         if call_module_cnt != 0:
#             print(f"Before inliner {gm.graph}")
#             inliner = _InlineModuleAndDecomp(gm)
#             # gm = inliner.transform()
#             import pdb; pdb.set_trace()
#             gm.graph.eliminate_dead_code()
#             gm.graph.lint()
#             gm.recompile()

#         afer_inline_call_module_cnt = 0
#         for n in gm.graph.nodes:
#             if n.op == 'call_module':
#                 afer_inline_call_module_cnt += 1

#         changed = call_module_cnt != afer_inline_call_module_cnt

#         print(f"After inliner {gm.graph}")
#         # if changed:
#             # import pdb;pdb.set_trace()

#         return changed