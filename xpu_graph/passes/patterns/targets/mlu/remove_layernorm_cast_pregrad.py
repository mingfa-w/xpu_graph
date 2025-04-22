import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import AutoMatchPattern


class RemoveLayerNormCastPregrad(AutoMatchPattern):
    _opt_level = OptLevel.level2
    # TODO(JYJ): Further experiments are needed to determine whether this pattern should be removed entirely or improved.
    #_support_stages = [FxStage.pregrad]
    def rewriter(self, gm: fx.GraphModule, rule_name: str, node_map: dict) -> bool:
        return False
        assert len(node_map) == 3
        pre_cast = node_map["pre_cast"]
        layernorm = node_map["layernorm"]
        post_cast = node_map["post_cast"]

        def _can_remove(pre_cast: fx.Node, post_cast: fx.Node) -> bool:
            inp = pre_cast.args[0]
            if inp.meta["tensor_meta"].dtype not in (
                torch.bfloat16,
                torch.float16,
            ):
                return False
            if pre_cast.kwargs["dtype"] != torch.float:
                return False
            if post_cast.kwargs["dtype"] != inp.meta["tensor_meta"].dtype:
                return False
            return True

        if not _can_remove(pre_cast, post_cast):
            return False

        layernorm.replace_input_with(layernorm.args[0], pre_cast.args[0])
        post_cast.replace_all_uses_with(layernorm)

        gm.graph.lint()
        gm.recompile()
        return True
