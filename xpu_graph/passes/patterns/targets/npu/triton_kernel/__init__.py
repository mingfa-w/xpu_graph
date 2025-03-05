

import torch
import torch_npu

from torch.library import Library, impl
npu_def = Library("torch_npu_triton", "DEF")

npu_lib = Library("npu_graph", "IMPL", "PrivateUse1")
npu_meta =Library("npu_graph", "IMPL", "Meta")

