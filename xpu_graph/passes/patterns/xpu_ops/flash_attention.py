import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.constant_manager import get_constant_manager

class FlashAttention(Pattern):
    def process(self, gm: fx.GraphModule):
        candidates = [node for node in gm.graph.nodes if node.op == 'call_function' and node.target == torch.ops.npu.npu_prompt_flash_attention.default]

        changed = False
        for fa_node in candidates:
            if 'pse_shift' in fa_node.kwargs:
                continue
            if 'input_layout' not in fa_node.kwargs or fa_node.kwargs['input_layout'] != 'BNSD':
                continue

            from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
            with maybe_disable_fake_tensor_mode():
                mask_constant = torch.tril(torch.ones(1024, 1024, dtype=torch.uint8).npu())
                mask_name = get_constant_manager(gm).register_constant(mask_constant,'casul_mask')

            with gm.graph.inserting_before(fa_node):
                mask = gm.graph.get_attr(
                    mask_name
                )
                xpu_fa_node = gm.graph.call_function(
                    torch.ops.xpu_ops.flash_attention.default,
                    args=(fa_node.args[0], fa_node.args[1], fa_node.args[2], mask, 0,),
                )
                fa_node.replace_all_uses_with(xpu_fa_node)
                gm.graph.erase_node(fa_node)
            changed = True

        gm.graph.lint()
        gm.recompile()
        return changed