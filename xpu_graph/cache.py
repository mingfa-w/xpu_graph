import os
import hashlib
import pickle
from os import PathLike
import torch
from torch._dynamo.convert_frame import compile_lock
from torch.utils._python_dispatch import _disable_current_modes
from .config import XpuGraphConfig
from .utils import logger

""" A base cache class does not store any thing"""


class XpuGraphCache:
    def cache_key(self, gm: torch.fx.GraphModule, fake_inputs, config: XpuGraphConfig):
        key = f"{gm}-{fake_inputs}-{config}"
        logger.info(f"Cache Key: \n{key}")
        hashkey = hashlib.md5(key.encode()).hexdigest()
        return hashkey

    def save_gm(
        self, key, value: torch.fx.GraphModule, expire=None
    ) -> torch.fx.GraphModule:
        # Note: since GraphModules ser/des may do canonicalization, so the cached version should be returned
        return value

    def load_gm(self, key):
        return None

    def delete_gm(self, key):
        return None


# 以下是一个简单的内存缓存实现示例
class XpuGraphLocalCache(XpuGraphCache):
    def __init__(self, cache_path: PathLike):
        super().__init__()
        cache_path = os.path.abspath(cache_path)
        os.makedirs(cache_path, exist_ok=True)
        self._path = cache_path

        # FIXME: Currently we manually set inductor cache dir for vendor compiler
        #        environs should not be tainted once AOT pipeline is ready
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_path, "inductor")

    def save_gm(
        self, key, value: torch.fx.GraphModule, expire=None
    ) -> torch.fx.GraphModule:
        artifact_path = self._graph_path(key)
        logger.info(f"Save cache in location: {artifact_path}")
        with compile_lock, _disable_current_modes():
            with open(artifact_path, "wb+") as f:
                pickle.dump(value, f)
            with open(artifact_path, "rb") as f:
                cached_graph = pickle.load(f)
        return cached_graph

    def load_gm(self, key):
        artifact_path = self._graph_path(key)
        if os.path.isfile(artifact_path):
            with compile_lock, _disable_current_modes():
                logger.info(f"Use cache in location: {artifact_path}")
                with open(artifact_path, "rb") as f:
                    cached_graph = pickle.load(f)
            return cached_graph
        else:
            return None

    def delete_gm(self, key):
        if key in self.cache:
            del self.cache[key]

    def _graph_path(self, key):
        fname = f"xpu_graph_{key}.pt"
        artifact_cache = os.path.join(self._path, fname)
        return artifact_cache


def default_cache():
    cache_path = os.getenv("XPU_GRAPH_CACHE_DIR")
    if cache_path is None:
        import tempfile
        cache_path = tempfile.mkdtemp(prefix="xpu_graph_")
        logger.debug(f"Use {cache_path} as default local cache")
    return XpuGraphLocalCache(cache_path)
