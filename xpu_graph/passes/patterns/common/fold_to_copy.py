# import torch
# import torch.fx as fx

# from xpu_graph.passes.patterns.pattern import Pattern

# class FoldToCopy(Pattern):
#     def process(self, gm: fx.GraphModule):
#         changed = False
#         candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target == torch.ops.aten._to_copy.default]

#         def _useless_to_copy(copy: fx.Node) -> bool:
#             # import pdb;pdb.set_trace()
#             inp = copy.args[0]
#             if not isinstance(inp, fx.Node):
#                 return False
#             if inp.meta['tensor_meta'].dtype != copy.meta['tensor_meta'].dtype:
#                 return False
#             if 'layout' in copy.kwargs:
#                 return False
#             import pdb;pdb.set_trace()
#             if inp.meta['val'].device != copy.meta['val'].device:
#                 return False
#             if 'pin_memory' in copy.kwargs or 'non_blocking' in copy.kwargs:
#                 return False
#             if inp.meta['tensor_meta'].memory_format != copy.meta['tensor_meta'].memory_format:
#                 return False
#             return True


#         for _to_copy in candidates:
#             if _useless_to_copy(_to_copy):
#                 changed=True
#                 _to_copy.replace_all_uses_with(_to_copy.args[0])
#                 gm.graph.erase_node(_to_copy)

#         gm.graph.lint()
#         gm.recompile()
#         return changed