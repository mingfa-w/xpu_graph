import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage


class PassManager:
    def __init__(self, config):

        from .optimizer import Optimizer

        Optimizer._debug = config.debug
        Optimizer._dump_graph = config.dump_graph

        self._passes = []
        from .patterns.pattern_manager import PatternManager

        self._pattern_manager = PatternManager(config)

        from .dce import Dce

        if Dce._opt_level <= config.opt_level:
            self._passes.append(Dce())

        from .cse import Cse

        if Cse._opt_level <= config.opt_level:
            self._passes.append(Cse())

        if config.constant_folding:
            from .constant_folding import ConstantFolding

            self._passes.append(ConstantFolding())

        self._passes.append(self._pattern_manager)

    def __call__(self, gm: fx.GraphModule, example_inputs, stage: FxStage):
        # Set pattern_manager to run stage-specific passes
        self.get_pattern_manager().set_stage(stage)
        changed = True
        while changed:
            from torch.fx.passes.shape_prop import ShapeProp
            ShapeProp(gm).propagate(*example_inputs)
            changed = False
            for pass_ in self._passes:
                changed = changed or pass_(gm)

        gm.recompile()

        # Note: Currently, we only inline modules with a E2E make_fx, just for serialize / desrialize
        from torch.fx.experimental.proxy_tensor import make_fx

        pre_dispatch = stage == FxStage.pregrad
        gm = make_fx(gm, pre_dispatch=pre_dispatch, record_module_stack=True)(
            *example_inputs
        )
        gm.delete_all_unused_submodules()
        gm.graph.lint()
        gm.recompile()
        return gm

    def get_pattern_manager(self):
        return self._pattern_manager
