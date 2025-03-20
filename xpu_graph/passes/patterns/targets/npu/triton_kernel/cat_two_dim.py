import torch
import torch_npu
import triton
import triton.language as tl
from typing import List

from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta


@triton.jit
def fn4_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, dim0_len: tl.constexpr,
             x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, 
             total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)


@triton.jit
def fn5_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, x4_ptr, dim0_len: tl.constexpr,
             x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, x4_len: tl.constexpr, 
             total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    x4 = tl.load(x4_ptr + tl.arange(0, dim0_len * x4_len))
    x4 = x4.reshape(dim0_len, x4_len)
    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)

    idx_start += x3_len
    nidx4 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x4_len)
    tl.store(output_ptr + nidx4, x4)


@triton.jit
def fn6_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, x4_ptr, x5_ptr, dim0_len: tl.constexpr,
             x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, x4_len: tl.constexpr, 
             x5_len: tl.constexpr, 
             total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    x4 = tl.load(x4_ptr + tl.arange(0, dim0_len * x4_len))
    x4 = x4.reshape(dim0_len, x4_len)
    x5 = tl.load(x5_ptr + tl.arange(0, dim0_len * x5_len))
    x5 = x5.reshape(dim0_len, x5_len)
    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)

    idx_start += x3_len
    nidx4 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x4_len)
    tl.store(output_ptr + nidx4, x4)

    idx_start += x4_len
    nidx5 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x5_len)
    tl.store(output_ptr + nidx5, x5)


@triton.jit
def fn7_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, x4_ptr, x5_ptr, x6_ptr, dim0_len: tl.constexpr,
             x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, x4_len: tl.constexpr, 
             x5_len: tl.constexpr, x6_len: tl.constexpr, 
             total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    x4 = tl.load(x4_ptr + tl.arange(0, dim0_len * x4_len))
    x4 = x4.reshape(dim0_len, x4_len)
    x5 = tl.load(x5_ptr + tl.arange(0, dim0_len * x5_len))
    x5 = x5.reshape(dim0_len, x5_len)
    x6 = tl.load(x6_ptr + tl.arange(0, dim0_len * x6_len))
    x6 = x6.reshape(dim0_len, x6_len)
    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)

    idx_start += x3_len
    nidx4 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x4_len)
    tl.store(output_ptr + nidx4, x4)

    idx_start += x4_len
    nidx5 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x5_len)
    tl.store(output_ptr + nidx5, x5)

    idx_start += x5_len
    nidx6 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x6_len)
    tl.store(output_ptr + nidx6, x6)


@triton.jit
def fn8_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, x4_ptr, x5_ptr, x6_ptr, x7_ptr, dim0_len: tl.constexpr,
             x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, x4_len: tl.constexpr, 
             x5_len: tl.constexpr, x6_len: tl.constexpr, x7_len: tl.constexpr, 
             total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    x4 = tl.load(x4_ptr + tl.arange(0, dim0_len * x4_len))
    x4 = x4.reshape(dim0_len, x4_len)
    x5 = tl.load(x5_ptr + tl.arange(0, dim0_len * x5_len))
    x5 = x5.reshape(dim0_len, x5_len)
    x6 = tl.load(x6_ptr + tl.arange(0, dim0_len * x6_len))
    x6 = x6.reshape(dim0_len, x6_len)
    x7 = tl.load(x7_ptr + tl.arange(0, dim0_len * x7_len))
    x7 = x7.reshape(dim0_len, x7_len)
    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)

    idx_start += x3_len
    nidx4 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x4_len)
    tl.store(output_ptr + nidx4, x4)

    idx_start += x4_len
    nidx5 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x5_len)
    tl.store(output_ptr + nidx5, x5)

    idx_start += x5_len
    nidx6 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x6_len)
    tl.store(output_ptr + nidx6, x6)

    idx_start += x6_len
    nidx7 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x7_len)
    tl.store(output_ptr + nidx7, x7)


@triton.jit
def fn9_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, x4_ptr, x5_ptr, x6_ptr, x7_ptr, x8_ptr, dim0_len: tl.constexpr,
             x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, x4_len: tl.constexpr, 
             x5_len: tl.constexpr, x6_len: tl.constexpr, x7_len: tl.constexpr, x8_len: tl.constexpr, 
             total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    x4 = tl.load(x4_ptr + tl.arange(0, dim0_len * x4_len))
    x4 = x4.reshape(dim0_len, x4_len)
    x5 = tl.load(x5_ptr + tl.arange(0, dim0_len * x5_len))
    x5 = x5.reshape(dim0_len, x5_len)
    x6 = tl.load(x6_ptr + tl.arange(0, dim0_len * x6_len))
    x6 = x6.reshape(dim0_len, x6_len)
    x7 = tl.load(x7_ptr + tl.arange(0, dim0_len * x7_len))
    x7 = x7.reshape(dim0_len, x7_len)
    x8 = tl.load(x8_ptr + tl.arange(0, dim0_len * x8_len))
    x8 = x7.reshape(dim0_len, x8_len)
    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)

    idx_start += x3_len
    nidx4 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x4_len)
    tl.store(output_ptr + nidx4, x4)

    idx_start += x4_len
    nidx5 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x5_len)
    tl.store(output_ptr + nidx5, x5)

    idx_start += x5_len
    nidx6 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x6_len)
    tl.store(output_ptr + nidx6, x6)

    idx_start += x6_len
    nidx7 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x7_len)
    tl.store(output_ptr + nidx7, x7)

    idx_start += x7_len
    nidx8 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x8_len)
    tl.store(output_ptr + nidx8, x8)


@triton.jit
def fn10_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, x4_ptr, 
              x5_ptr, x6_ptr, x7_ptr, x8_ptr, x9_ptr, 
              dim0_len: tl.constexpr,
              x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, x4_len: tl.constexpr, 
              x5_len: tl.constexpr, x6_len: tl.constexpr, x7_len: tl.constexpr, x8_len: tl.constexpr, x9_len: tl.constexpr,
              total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    x4 = tl.load(x4_ptr + tl.arange(0, dim0_len * x4_len))
    x4 = x4.reshape(dim0_len, x4_len)

    x5 = tl.load(x5_ptr + tl.arange(0, dim0_len * x5_len))
    x5 = x5.reshape(dim0_len, x5_len)
    x6 = tl.load(x6_ptr + tl.arange(0, dim0_len * x6_len))
    x6 = x6.reshape(dim0_len, x6_len)
    x7 = tl.load(x7_ptr + tl.arange(0, dim0_len * x7_len))
    x7 = x7.reshape(dim0_len, x7_len)
    x8 = tl.load(x8_ptr + tl.arange(0, dim0_len * x8_len))
    x8 = x8.reshape(dim0_len, x8_len)
    x9 = tl.load(x9_ptr + tl.arange(0, dim0_len * x9_len))
    x9 = x9.reshape(dim0_len, x9_len)

    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)

    idx_start += x3_len
    nidx4 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x4_len)
    tl.store(output_ptr + nidx4, x4)
    # ---
    idx_start += x4_len
    nidx5 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x5_len)
    tl.store(output_ptr + nidx5, x5)
    
    idx_start += x5_len
    nidx6 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x6_len)
    tl.store(output_ptr + nidx6, x6)

    idx_start += x6_len
    nidx7 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x7_len)
    tl.store(output_ptr + nidx7, x7)

    idx_start += x7_len
    nidx8 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x8_len)
    tl.store(output_ptr + nidx8, x8)

    idx_start += x8_len
    nidx9 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x9_len)
    tl.store(output_ptr + nidx9, x9)


@triton.jit
def fn11_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, x4_ptr, 
              x5_ptr, x6_ptr, x7_ptr, x8_ptr, x9_ptr, x10_ptr,
              dim0_len: tl.constexpr,
              x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, x4_len: tl.constexpr, 
              x5_len: tl.constexpr, x6_len: tl.constexpr, x7_len: tl.constexpr, x8_len: tl.constexpr, x9_len: tl.constexpr, x10_len: tl.constexpr,
              total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    x4 = tl.load(x4_ptr + tl.arange(0, dim0_len * x4_len))
    x4 = x4.reshape(dim0_len, x4_len)
    x5 = tl.load(x5_ptr + tl.arange(0, dim0_len * x5_len))
    x5 = x5.reshape(dim0_len, x5_len)
    x6 = tl.load(x6_ptr + tl.arange(0, dim0_len * x6_len))
    x6 = x6.reshape(dim0_len, x6_len)
    x7 = tl.load(x7_ptr + tl.arange(0, dim0_len * x7_len))
    x7 = x7.reshape(dim0_len, x7_len)
    x8 = tl.load(x8_ptr + tl.arange(0, dim0_len * x8_len))
    x8 = x8.reshape(dim0_len, x8_len)
    x9 = tl.load(x9_ptr + tl.arange(0, dim0_len * x9_len))
    x9 = x9.reshape(dim0_len, x9_len)
    x10 = tl.load(x10_ptr + tl.arange(0, dim0_len * x10_len))
    x10 = x10.reshape(dim0_len, x10_len)

    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)

    idx_start += x3_len
    nidx4 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x4_len)
    tl.store(output_ptr + nidx4, x4)
    # ---
    idx_start += x4_len
    nidx5 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x5_len)
    tl.store(output_ptr + nidx5, x5)
    
    idx_start += x5_len
    nidx6 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x6_len)
    tl.store(output_ptr + nidx6, x6)

    idx_start += x6_len
    nidx7 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x7_len)
    tl.store(output_ptr + nidx7, x7)

    idx_start += x7_len
    nidx8 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x8_len)
    tl.store(output_ptr + nidx8, x8)

    idx_start += x8_len
    nidx9 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x9_len)
    tl.store(output_ptr + nidx9, x9)

    idx_start += x9_len
    nidx10 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x10_len)
    tl.store(output_ptr + nidx10, x10)



@triton.jit
def fn12_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, x4_ptr, 
              x5_ptr, x6_ptr, x7_ptr, x8_ptr, x9_ptr, 
              x10_ptr, x11_ptr,
              dim0_len: tl.constexpr,
              x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr, x4_len: tl.constexpr, 
              x5_len: tl.constexpr, x6_len: tl.constexpr, x7_len: tl.constexpr, x8_len: tl.constexpr, x9_len: tl.constexpr, 
              x10_len: tl.constexpr, x11_len: tl.constexpr,
              total_dim1_len: tl.constexpr):
    
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    x4 = tl.load(x4_ptr + tl.arange(0, dim0_len * x4_len))
    x4 = x4.reshape(dim0_len, x4_len)
    x5 = tl.load(x5_ptr + tl.arange(0, dim0_len * x5_len))
    x5 = x5.reshape(dim0_len, x5_len)
    x6 = tl.load(x6_ptr + tl.arange(0, dim0_len * x6_len))
    x6 = x6.reshape(dim0_len, x6_len)
    x7 = tl.load(x7_ptr + tl.arange(0, dim0_len * x7_len))
    x7 = x7.reshape(dim0_len, x7_len)
    x8 = tl.load(x8_ptr + tl.arange(0, dim0_len * x8_len))
    x8 = x8.reshape(dim0_len, x8_len)
    x9 = tl.load(x9_ptr + tl.arange(0, dim0_len * x9_len))
    x9 = x9.reshape(dim0_len, x9_len)
    x10 = tl.load(x10_ptr + tl.arange(0, dim0_len * x10_len))
    x10 = x10.reshape(dim0_len, x10_len)
    x11 = tl.load(x11_ptr + tl.arange(0, dim0_len * x11_len))
    x11 = x11.reshape(dim0_len, x11_len)

    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)

    idx_start += x3_len
    nidx4 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x4_len)
    tl.store(output_ptr + nidx4, x4)
    # ---
    idx_start += x4_len
    nidx5 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x5_len)
    tl.store(output_ptr + nidx5, x5)
    
    idx_start += x5_len
    nidx6 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x6_len)
    tl.store(output_ptr + nidx6, x6)

    idx_start += x6_len
    nidx7 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x7_len)
    tl.store(output_ptr + nidx7, x7)

    idx_start += x7_len
    nidx8 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x8_len)
    tl.store(output_ptr + nidx8, x8)

    idx_start += x8_len
    nidx9 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x9_len)
    tl.store(output_ptr + nidx9, x9)

    idx_start += x9_len
    nidx10 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x10_len)
    tl.store(output_ptr + nidx10, x10)

    idx_start += x10_len
    nidx11 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x11_len)
    tl.store(output_ptr + nidx11, x11)


def get_total_dim_len(input_tensors, dim):
    total_len = 0
    if dim == -1:
        dim = 1
    for ten in input_tensors:
        total_len += ten.shape[dim]
    return total_len



npu_def.define("cat_two_dim(Tensor[] input_tensors, int dim) -> (Tensor)")
@impl(npu_lib, "cat_two_dim")
def cat_two_dim(
    input_tensors: List[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    inp_num = len(input_tensors)
    #if dim == 1 or dim == -1:
    total_dim1_len = get_total_dim_len(input_tensors, 1)
    dim0_len = input_tensors[0].shape[0]
    output_tensor = torch.zeros((dim0_len, total_dim1_len), dtype=input_tensors[0].dtype, device=input_tensors[0].device)
    if  inp_num == 4:
        fn4_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], 
                        dim0_len, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], input_tensors[3].shape[1], 
                        total_dim1_len)
        return output_tensor
    elif  inp_num == 5:
        fn5_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4],
                        dim0_len, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], input_tensors[3].shape[1], input_tensors[4].shape[1], 
                        total_dim1_len)
        return output_tensor
    elif  inp_num == 6:
        fn6_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4],
                        input_tensors[5],
                        dim0_len, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], input_tensors[3].shape[1], input_tensors[4].shape[1], 
                        input_tensors[5].shape[1], 
                        total_dim1_len)
        return output_tensor
    elif  inp_num == 7:
        fn7_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4],
                        input_tensors[5], input_tensors[6],
                        dim0_len, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], input_tensors[3].shape[1], input_tensors[4].shape[1],
                        input_tensors[5].shape[1], input_tensors[6].shape[1],
                        total_dim1_len)
        return output_tensor
    elif  inp_num == 8:
        fn8_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4],
                        input_tensors[5], input_tensors[6], input_tensors[7],
                        dim0_len, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], input_tensors[3].shape[1], input_tensors[4].shape[1], 
                        input_tensors[5].shape[1], input_tensors[6].shape[1], input_tensors[7].shape[1], 
                        total_dim1_len)
        return output_tensor
    elif  inp_num == 9:
        fn9_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4],
                        input_tensors[5], input_tensors[6], input_tensors[7], input_tensors[8],
                        dim0_len, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], input_tensors[3].shape[1], input_tensors[4].shape[1], 
                        input_tensors[5].shape[1], input_tensors[6].shape[1], input_tensors[7].shape[1],  input_tensors[8].shape[1], 
                        total_dim1_len)
        return output_tensor
    elif inp_num == 10:
        fn10_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4],
                        input_tensors[5], input_tensors[6], input_tensors[7], input_tensors[8], input_tensors[9],
                        86, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], 
                        input_tensors[3].shape[1], input_tensors[4].shape[1], input_tensors[5].shape[1], 
                        input_tensors[6].shape[1], input_tensors[7].shape[1], input_tensors[8].shape[1], 
                        input_tensors[9].shape[1],
                        total_dim1_len)
        return output_tensor
    elif inp_num == 11:
        fn11_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4],
                        input_tensors[5], input_tensors[6], input_tensors[7], input_tensors[8], input_tensors[9], 
                        input_tensors[10],
                        86, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], 
                        input_tensors[3].shape[1], input_tensors[4].shape[1], input_tensors[5].shape[1], 
                        input_tensors[6].shape[1], input_tensors[7].shape[1], input_tensors[8].shape[1], 
                        input_tensors[9].shape[1], input_tensors[10].shape[1],
                        total_dim1_len)
        return output_tensor
    elif inp_num == 12:
        fn12_dim1[(1,1,1)](output_tensor, 
                        input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4],
                        input_tensors[5], input_tensors[6], input_tensors[7], input_tensors[8], input_tensors[9], 
                        input_tensors[10], input_tensors[11],
                        86, 
                        input_tensors[0].shape[1], input_tensors[1].shape[1], input_tensors[2].shape[1], 
                        input_tensors[3].shape[1], input_tensors[4].shape[1], input_tensors[5].shape[1], 
                        input_tensors[6].shape[1], input_tensors[7].shape[1], input_tensors[8].shape[1], 
                        input_tensors[9].shape[1], input_tensors[10].shape[1], input_tensors[11].shape[1],
                        total_dim1_len)
        return output_tensor
    else:
        # no implementation
        return output_tensor
        


@impl(npu_meta, "cat_two_dim")
def cat_two_dim_fake(
    input_tensors: List[torch.Tensor],
    dim: int,
):
    dim0_len = 0
    total_dim1_len = 0
    dim0_len = input_tensors[0].shape[0]
    total_dim1_len = get_total_dim_len(input_tensors, 1)
    new_shape = (dim0_len, total_dim1_len)
    output = torch.empty(new_shape, device=input_tensors[0].device, dtype=input_tensors[0].dtype)
    return output