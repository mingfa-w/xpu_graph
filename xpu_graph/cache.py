import os
import hashlib
import dill
from os import PathLike

import torch
from torch._dynamo.convert_frame import compile_lock
from torch.utils._python_dispatch import _disable_current_modes
from torch.fx import GraphModule, Graph, Node, map_arg

from .config import XpuGraphConfig
from .utils import logger
from .fx_utils import FxStage


class XpuGraphCache:
    """A base cache class does not store any thing"""

    def cache_key(
        self,
        gm: torch.fx.GraphModule,
        fake_inputs,
        config: XpuGraphConfig,
        stage: FxStage,
    ):
        key = f"{gm}-{fake_inputs}-{config}-{stage}"
        logger.debug(f"Cache Key readable: \n{key}")
        hashkey = hashlib.md5(key.encode()).hexdigest()
        logger.info(f"Cache Key: {hashkey}")
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

        serialize_gm(value, artifact_path)
        cached_graph = deserialize_gm(artifact_path)

        return cached_graph

    def load_gm(self, key):
        artifact_path = self._graph_path(key)
        if os.path.isfile(artifact_path):
            logger.info(f"Use cache in location: {artifact_path}")
            cached_graph = deserialize_gm(artifact_path)
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


def no_cache():
    return XpuGraphCache()


def default_cache():
    cache_path = os.getenv("XPU_GRAPH_CACHE_DIR")
    if cache_path is None:
        import tempfile

        cache_path = tempfile.mkdtemp(prefix="xpu_graph_")
        logger.debug(f"Use {cache_path} as default local cache")
    return XpuGraphLocalCache(cache_path)


class _GraphModuleData:
    def __init__(self, gm: GraphModule):
        if len(gm._modules) > 0:
            raise NotImplemented("Only fully-inlined graph module is supported")
        self.gm_dict = gm.__dict__.copy()
        del self.gm_dict["_graph"]
        self.graph = _GraphData(gm.graph)

    def unpickle(self) -> GraphModule:
        gm = GraphModule.__new__(GraphModule)
        gm.__dict__ = self.gm_dict
        gm.graph = self.graph.unpickle(gm)
        gm.recompile()
        return gm


class _GraphData:
    def __init__(self, graph: Graph):
        self.nodes = [_NodeData(node) for node in graph.nodes]
        self.tracer_cls = graph._tracer_cls
        self.tracer_extras = graph._tracer_extras

    def unpickle(self, root_mod: GraphModule) -> Graph:
        graph = Graph(root_mod, self.tracer_cls, self.tracer_extras)
        node_dict = {}
        for node in self.nodes:
            n = node.unpickle(graph, node_dict)
            node_dict[n.name] = n
        return graph


class _NodeData:
    def __init__(self, node: Node):
        self.name = node.name
        self.type = node.type
        self.op = node.op
        self.target = node._pretty_print_target(node.target)

        self.args = tuple(map_arg(node.args, lambda n: _ArgWrapper(n)))
        self.kwargs = dict(map_arg(node.kwargs, lambda n: _ArgWrapper(n)))

    def unpickle(self, graph: Graph, node_dict):
        def _unwrap(arg):
            if isinstance(arg, _ArgWrapper):
                return node_dict[arg.name]
            else:
                return arg

        from torch.fx.node import map_aggregate

        args = map_aggregate(self.args, _unwrap)
        kwargs = map_aggregate(self.kwargs, _unwrap)
        if self.op == "call_function":
            target = _get_target_function(self.target)
        else:
            target = self.target
        return graph.create_node(self.op, target, args, kwargs, self.name, self.type)


def _get_target_function(fn_name: str):
    fqn_list = fn_name.split(".")
    import torch
    import operator
    import builtins

    supported_mods = {"torch": torch, "operator": operator, "builtins": builtins}
    try:
        target = supported_mods[fqn_list[0]]
        for attr in fqn_list[1:]:
            target = getattr(target, attr)
        assert callable(target)
    except:
        raise NotImplementedError(f"Unsupported call_function: {fn_name}")
    return target


class _ArgWrapper:
    def __init__(self, node: Node):
        self.name = node.name


def serialize_gm(gm: GraphModule, path):
    with compile_lock, _disable_current_modes():
        gm_data = _GraphModuleData(gm)
        with open(path, "wb+") as f:
            dill.dump(gm_data, f)


def deserialize_gm(path):
    with compile_lock, _disable_current_modes():
        with open(path, "rb") as f:
            gm_data = dill.load(f)
            gm = gm_data.unpickle()
    return gm
