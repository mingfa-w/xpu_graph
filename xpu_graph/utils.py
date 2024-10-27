import logging
import sys
import time

logger = logging.getLogger("XPU_GRAPH")

def setup_logger(loglevel):
    while logger.hasHandlers():
        return
    fmt = logging.Formatter(
        fmt="[XPU_GRAPH]: %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    logger.propagate = False


def xpu_timer(func):
    def wrapper(*args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            start = time.time()
            res= func(*args, **kwargs)
            end = time.time()
            logger.debug(f"{class_name}.{func.__name__} cost {end - start}s")
        else:
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            logger.debug(f"{func.__name__} cost {end - start}s")
        return res

    return wrapper
