import pkgutil
import importlib

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

def get_all_patterns(opt_level: int):
    patterns = []

    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_name}")

        for name in dir(module):

            pat = getattr(module, name)
            if isinstance(pat, type) and issubclass(pat, Pattern) and pat != Pattern and pat._opt_level <= opt_level:
                patterns.append(pat())

    logger.debug(f"xpu_graph enable builtin xpu_ops optimizers: {[pat.__class__.__name__ for pat in patterns]}")

    return patterns