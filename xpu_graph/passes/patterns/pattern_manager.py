from typing import Callable, overload

import torch
import torch.fx as fx

from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.config import XpuGraphConfig, Target
from .pattern import Pattern
from xpu_graph.utils import logger

class PatternManager(Optimizer):
    def __init__(self, config: XpuGraphConfig):
        super().__init__()

        self._patterns = []

        from .common import get_all_patterns as get_common_patterns
        self._patterns += get_common_patterns(config)

        from .structure import get_all_patterns as get_structure_patterns
        self._patterns += get_structure_patterns(config)

        if config.use_xpu_ops:
            from .xpu_ops import get_all_patterns as get_xpu_ops_patterns
            self._patterns += get_xpu_ops_patterns(config)

        from .targets import get_all_patterns as get_target_patterns
        self._patterns += get_target_patterns(config)


    def process(self, gm: fx.GraphModule) -> bool:
        changed = False
        for pattern in self._patterns:
            changed = changed or pattern(gm)

        return changed


    @overload
    def register_pattern(self, pattern: Pattern):
        ...

    @overload
    def register_pattern(self, matcher: Callable, replacement: Callable):
        ...

    def register_pattern(self, *args):
        if len(args) == 1:
            self._patterns.append(args[0]())
        elif len(args) == 2:
            class _Pattern(Pattern):
                def __init__(self):
                    super().__init__()

                def __call__(self, gm: fx.GraphModule) -> bool:
                    from torch.fx import subgraph_rewriter
                    match = subgraph_rewriter.replace_pattern(gm, args[0], args[1])

                    return len(match)

            self._patterns.append(_Pattern())
