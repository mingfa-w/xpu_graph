import torch
import torch.fx
from collections import defaultdict


def find_module(gm, module_name):
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == module_name:
            return True
    return False


def get_module_name(gm, module_name):
    name_counter = 0
    while True:
        new_module_name = module_name + "_" + str(name_counter)
        if not find_module(gm, new_module_name):
            return new_module_name
        else:
            name_counter += 1
