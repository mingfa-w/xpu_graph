import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern

class AlignMMWeight(Pattern):

    def process(self, gm: fx.GraphModule):
        changed = False
        graph = gm.graph
        mm_nodes = [node for node in graph.nodes if node.op == 'call_function' and node.target == torch.ops.xpu_ops.quant_matmul.default]

        for node in mm_nodes:
            if node.args[1].op != 'get_attr':
                continue
            if len(node.args[1].users) != 1:
                continue

            weight_shape = node.args[1].meta['tensor_meta'].shape
            if weight_shape[0] % 512 == 0 and weight_shape[1] % 512 != 0:
                changed = True

                from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
                with maybe_disable_fake_tensor_mode():
                    weight = getattr(gm, node.args[1].target)

                    new_weight = weight.T.contiguous()

                    from xpu_graph.constant_manager import get_constant_manager
                    new_weight_name = get_constant_manager(gm).register_constant(new_weight, node.args[1].target + "_transposed")

                with gm.graph.inserting_before(node):
                    new_weight_op = gm.graph.get_attr(new_weight_name)
                    node.update_arg(1, new_weight_op)
                    node.update_arg(7, not node.args[7])

                gm.graph.lint()
                gm.recompile()

        return changed
