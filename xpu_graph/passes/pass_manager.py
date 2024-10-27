import torch
import torch.fx as fx

# TODO: Add a config to make user config this PassManager
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

        from .constant_folding import ConstantFolding
        if ConstantFolding._opt_level <= config.opt_level:
            self._passes.append(ConstantFolding())

        self._passes.append(self._pattern_manager)

    def __call__(self, gm: fx.GraphModule, example_inputs):
        changed = True
        while changed:
            changed = False
            for pass_ in self._passes:
                changed = changed or pass_(gm)

            from torch.fx.passes.shape_prop import ShapeProp
            ShapeProp(gm).propagate(*example_inputs)

        gm.recompile()
        return gm

    def get_pattern_manager(self):
        return self._pattern_manager