import torch
import torch.fx as fx
import numpy as np
import hashlib

from xpu_graph.fx_utils import get_disable_fake_mode_handler
from xpu_graph.utils import logger

constant_manager_map = {}


class ConstantManager:
    def __init__(self, gm: fx.GraphModule):
        self._constant_id = 0
        self._gm = gm
        self._constants_keeper = {}

        for n in gm.graph.nodes:
            if n.op == "get_attr":
                assert hasattr(gm, n.target)
                constant = getattr(gm, n.target)
                real_name = self.register_constant(constant, n.target)
                n.target = real_name

    def _get_constant_hash(self, constant: torch.Tensor):
        disable_fake_mode = get_disable_fake_mode_handler()
        with disable_fake_mode():
            cpu_tensor = constant.detach().cpu()

            if not cpu_tensor.is_contiguous():
                cpu_tensor = cpu_tensor.contiguous()

            raw_data = cpu_tensor.numpy().tobytes()
            meta = (
                str(constant.shape).encode("utf-8")
                + str(constant.stride()).encode("utf-8")
                + str(constant.dtype).encode("utf-8")
                + str(constant.device).encode("utf-8")
            )
            hasher = hashlib.md5()
            hasher.update(meta)
            hasher.update(raw_data)

            return hasher.hexdigest()

    def register_constant(self, constant: torch.Tensor, name: str) -> str:
        """
        Register a constant folding result.
        """
        if hasattr(self._gm, name):
            constant_name = name
        else:
            constant_name = name + f"_{self._constant_id}"
            self._constant_id += 1
            logger.debug(f"register constant: {constant_name}")
            self._gm.register_buffer(constant_name, constant)

        # constant_name = name
        constant_hash = self._get_constant_hash(constant)
        logger.debug(f"constant_name: {constant_name}, constant_hash: {constant_hash}")
        if constant_hash not in self._constants_keeper:
            self._constants_keeper[constant_hash] = [constant_name]
        else:
            self._constants_keeper[constant_hash].append(constant_name)

        logger.debug(f"constants_keeper: {self._constants_keeper}")

        return constant_name

    # def remove_useless_constants(self) -> bool:
    #     # Remove useless module's parameters and buffers
    #     used_constant_names = set()
    #     for n in self._gm.graph.nodes:
    #         if n.op == "get_attr":
    #             used_constant_names.add(n.target)

    #     should_remove_parameters = [
    #         name
    #         for name in self._gm._parameters.keys()
    #         if name not in used_constant_names
    #     ]
    #     for name in should_remove_parameters:
    #         logger.debug(f"remove parameter: {name}")
    #         delattr(self._gm, name)

    #     should_remove_buffers = [
    #         name for name in self._gm._buffers.keys() if name not in used_constant_names
    #     ]
    #     for name in should_remove_buffers:
    #         logger.debug(f"remove buffer: {name}")
    #         delattr(self._gm, name)

    #     # Update constant_keeper
    #     should_remove_keeper = [
    #         hash
    #         for hash, name in self._constants_keeper.items()
    #         if name in should_remove_parameters + should_remove_buffers
    #     ]
    #     for hash in should_remove_keeper:
    #         del self._constants_keeper[hash]

    #     logger.debug(f"Keep constants: {self._constants_keeper.values()}")
    #     logger.debug(f"module parameters: {self._gm._parameters.keys()}")
    #     logger.debug(f"module buffers: {self._gm._buffers.keys()}")

    def remove_redundant_constants(self) -> bool:
        changed = False
        get_attr_nodes = {
            node.target: node for node in self._gm.graph.nodes if node.op == "get_attr"
        }

        all_names = []
        for name_list in self._constants_keeper.values():
            for i, name in enumerate(name_list):
                all_names.append(name)
                if i == 0:
                    continue
                changed = True
                get_attr_nodes[name].target = name_list[0]

        used_constant_names = set(
            node.target for node in self._gm.graph.nodes if node.op == "get_attr"
        )

        for hash, name_list in self._constants_keeper.items():
            new_name_list = []
            for i, name in enumerate(name_list):
                if name not in used_constant_names:
                    changed = True
                    delattr(self._gm, name)
                else:
                    new_name_list.append(name)
            self._constants_keeper[hash] = new_name_list

        logger.debug(f"Keep constants: {self._constants_keeper.values()}")
        logger.debug(f"module parameters: {self._gm._parameters.keys()}")
        logger.debug(f"module buffers: {self._gm._buffers.keys()}")

        return changed


def get_constant_manager(gm):
    if gm not in constant_manager_map:
        constant_manager_map[gm] = ConstantManager(gm)

    return constant_manager_map[gm]
