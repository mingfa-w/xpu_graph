import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

class Multiflow():
    FlowNum = os.getenv('NPU_FLOWNUM', 1)
    AicNum = os.getenv('NPU_AICNUM', 24)
    AivNum = 2 * AicNum

class Target(Enum):
    ascend = "ascend"
    mlu = "mlu"
    npu = "npu"
    none = "none"


class OptLevel(Enum):
    level0 = 0
    level1 = 1
    level2 = 2
    level3 = 3

    def __lt__(self, other):
        if isinstance(other, OptLevel):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, OptLevel):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OptLevel):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, OptLevel):
            return self.value >= other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, OptLevel):
            return self.value == other.value
        return NotImplemented


class ExecuteMode(Enum):
    eager = "eager"
    graph = "graph"


@dataclass
class XpuGraphConfig:
    debug: bool = False
    target: Target = field(default=Target.none)
    opt_level: OptLevel = OptLevel.level1
    dump_graph: bool = False
    enable_cache: bool = True
    use_xpu_ops: bool = False  # Use xpu_ops or not
    freeze: bool = False  # Freeze parameter, will do better constant_folding
    constant_folding: bool = False
    skip_all_pass: bool = (
        False  # Default false, use for debug, which will skip all passes of xpu_ops
    )
    # Mode: {"cudagraphs", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
    # we add a "cudagraphs" option. At this mode, only torch.compile in-tree backend cudugraphs will be enable.
    # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
    vendor_compiler_config: Optional[Dict[str, Any]] = None
