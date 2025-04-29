import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.utils import logger


class PassManager:
    def __init__(self, config):
        self._config = config
        self._stage = FxStage.inference
        self._passes = []
        self._enable_passes = []

        from .optimizer import Optimizer

        Optimizer._debug = self._config.debug
        Optimizer._dump_graph = self._config.dump_graph

        from .patterns.pattern_manager import PatternManager

        self._pattern_manager = PatternManager(self._config)

        from .dce import Dce

        if Dce._opt_level <= self._config.opt_level:
            self._passes.append(Dce())

        from .cse import Cse

        # FIXME(zhangjihang): CSE will introduce some accurancy problem during pregrad stage, so I just skip it for safety now.
        if Cse._opt_level <= self._config.opt_level:
            self._passes.append(Cse())

        if self._config.constant_folding:
            from .constant_folding import ConstantFolding

            self._passes.append(ConstantFolding())

        self._passes.append(self._pattern_manager)

    def reset_enable_passes_with_stage(self, stage: FxStage):
        self._enable_passes = []
        for pass_ in self._passes:
            enable_pass = pass_.get_pass_with_stage(stage)
            if enable_pass:
                self._enable_passes.append(enable_pass)

    def __call__(self, gm: fx.GraphModule, example_inputs, stage: FxStage):
        # Set pattern_manager to run stage-specific passes
        self.reset_enable_passes_with_stage(stage)

        changed = True
        while changed:
            from xpu_graph.passes.fake_tensor_prop import FakeTensorProp
            from torch._guards import active_fake_mode

            FakeTensorProp(gm, active_fake_mode()).propagate(*example_inputs)

            changed = False
            for optimizer in self._enable_passes:
                changed = changed or optimizer(gm)

        return gm

    def get_pattern_manager(self):
        return self._pattern_manager
