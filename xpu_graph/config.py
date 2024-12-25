from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class Target(Enum):
    ascend = "ascend"
    mlu = "mlu"
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
    use_xpu_ops: bool = False  # Use xpu_ops or not
    freeze: bool = False  # Freeze parameter, will do better constant_folding
    vendor_compiler: Optional[Dict[str, Any]] = None
