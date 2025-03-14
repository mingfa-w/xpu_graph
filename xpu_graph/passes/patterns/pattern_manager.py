from typing import Callable, overload, List

import torch
import torch.fx as fx

from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.config import XpuGraphConfig, Target
from xpu_graph.fx_utils import FxStage
from .pattern import Pattern, PatternGroup


class PatternManager(Optimizer):
    def __init__(self, config: XpuGraphConfig):
        super().__init__()

        self._patterns = {
            PatternGroup.GROUP0: [],
            PatternGroup.GROUP1: [],
            PatternGroup.GROUP2: [],
        }

        from .common import get_all_patterns as get_common_patterns

        for group, patterns in get_common_patterns(config).items():
            self._patterns[group] += patterns

        from .structure import get_all_patterns as get_structure_patterns

        for group, patterns in get_structure_patterns(config).items():
            self._patterns[group] += patterns

        if config.use_xpu_ops:
            from .xpu_ops import get_all_patterns as get_xpu_ops_patterns

            for group, patterns in get_xpu_ops_patterns(config).items():
                self._patterns[group] += patterns

        from .targets import get_all_patterns as get_target_patterns

        for group, patterns in get_target_patterns(config).items():
            self._patterns[group] += patterns

        self._stage = FxStage.inference

    def set_stage(self, stage: FxStage):
        prev_stage = self._stage
        self._stage = stage
        return prev_stage

    def process(self, gm: fx.GraphModule) -> bool:
        changed = False
        loop_time = 5
        for group in sorted(self._patterns.keys()):
            for i in range(loop_time):
                for pattern in self._patterns[group]:
                    if self._stage == pattern._stage:
                        changed = changed or pattern(gm)

        return changed

    @overload
    def register_pattern(self, pattern: Pattern): ...

    @overload
    def register_pattern(self, matcher: Callable, replacement: Callable): ...

    @overload
    def register_pattern(
        self, matcher: Callable, replacement: Callable, stage: FxStage
    ): ...

    def register_pattern(self, *args):
        if len(args) == 1:
            pat = args[0]
            for stage in pat._stages:
                self._patterns[pat._pattern_group].append(pat(stage))
        elif len(args) == 2 or len(args) == 3:

            class _Pattern(Pattern):
                def __init__(self, stage):
                    super().__init__(stage)

                def __call__(self, gm: fx.GraphModule) -> bool:
                    from torch.fx import subgraph_rewriter

                    match = subgraph_rewriter.replace_pattern(gm, args[0], args[1])

                    return len(match)

            if len(args) == 3:
                _Pattern._stages = [args[2]]
            for stage in _Pattern._stages:
                self._patterns[_Pattern._pattern_group].append(_Pattern(stage))
