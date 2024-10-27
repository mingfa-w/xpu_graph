import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern

class QuantMatmul(Pattern):
    def process(self, gm: fx.GraphModule):
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target == torch.ops.npu.npu_quant_matmul.default]

        changed = False
        for qm_node in candidates:
            if 'offset' in qm_node.kwargs:
                continue
            if qm_node.kwargs['output_dtype'] != torch.bfloat16:
                continue

            from typing import Tuple
            def _flatten_transpose_input(node: fx.Node) -> Tuple[fx.Node, bool]:
                if node.op == 'call_function' and node.target == torch.ops.aten.t.default:
                    return node.args[0], True
                return node, False

            inp0, inp0_transpose = _flatten_transpose_input(qm_node.args[0])
            inp1, inp1_transpose = _flatten_transpose_input(qm_node.args[1])

            changed = True
            with gm.graph.inserting_before(qm_node):
                xpu_qm_node = gm.graph.call_function(
                    torch.ops.xpu_ops.quant_matmul.default,
                    args=(
                        inp0,
                        inp1,
                        qm_node.args[2],
                        qm_node.kwargs['bias'],
                        qm_node.kwargs['pertoken_scale'],
                        'bfloat16',
                        inp0_transpose,
                        inp1_transpose,
                    )
                )
                qm_node.replace_all_uses_with(xpu_qm_node)
                gm.graph.erase_node(qm_node)

        gm.graph.lint()
        gm.recompile()
        return changed