import pkgutil
import importlib

from xpu_graph.passes.patterns.pattern import Pattern, AutoMatchPattern
from xpu_graph.config import XpuGraphConfig, Target
from xpu_graph.utils import logger


def get_all_patterns(config: XpuGraphConfig):
    patterns = []

    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_name}")

        structure_preplacements = []
        if config.target == Target.mlu:
            from ..targets.mlu.structure_replacements import get_structure_replacements
            structure_preplacements = get_structure_replacements()

        for name in dir(module):
            pat = getattr(module, name)
            if (
                isinstance(pat, type)
                and issubclass(pat, Pattern)
                and pat not in (Pattern, AutoMatchPattern)
                and pat._opt_level <= config.opt_level
            ):
                if pat.__name__ in structure_preplacements:
                    patterns.append(pat(structure_preplacements[pat.__name__]))

    logger.debug(
        f"xpu_graph enable builtin structure patterns: {[pat.__class__.__name__ for pat in patterns]}"
    )

    return patterns
