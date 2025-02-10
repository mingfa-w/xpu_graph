"""
Fused Attention
===============
  
This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team
  
Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
  
"""
  
import pytest
import torch
import torch_mlu
  
import triton
import numpy as np
#from genesis.Python.Test.Language.utils import *
import triton.language as tl
from triton.language.extra.mlu.libdevice import ultra_silu_float2bf16
from triton.language.extra.mlu.libdevice import ultra_silu_float2half
from triton.language.extra.mlu.libdevice import ultra_silu
#from triton.language.extra.mlu.libdevice import cycle_sub_exp
  
IS_Y = False
 
@triton.jit
def silu_forward(x, PERF_OPT: tl.constexpr, IS_Y: tl.constexpr):
    x_fp32 = x.to(tl.float32)
    y = None
    if PERF_OPT == 0:
        y = ultra_silu_float2bf16(x_fp32) * 0.0002450379808870375
        #y = ultra_silu_float2half(x_fp32)
        #tl.device_print("y: ", x_fp32)
        #y = x_fp32
    elif PERF_OPT == 1:
        y = ultra_silu(x_fp32) * 0.0002450379808870375
    # PERF_OPT == 2
    else:
        if IS_Y:
            # for y, use tanh
            y = 0.5 * x_fp32
            t = tl.tanh(y)
            y = (t + 1) * y
        else:
            log2e = -1.442695041
            #log2e = 0xbfb8aa3b
            #log2e = log2e.to(tl.float32, bitcast=True)
            #y = 1.0 / (1.0 + cycle_sub_exp(x_fp32, 0, log2e)) * x_fp32
            # No instruction fusion
            y = 1.0 / (1.0 + tl.exp2(x_fp32 * log2e)) * x_fp32 * 0.0002450379808870375
            #y = 1.0 / (1.0 + tl.exp2(x_fp32 * log2e)) * x_fp32
    return y
  
@triton.jit
def _attn_fwd_inner(
        off_z,
        acc,
        q,  #
        BIAS,
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,  #
        CAUSAL: tl.constexpr,
        offs_m: tl.constexpr,
        offs_n: tl.constexpr,  #
        N_CTX: tl.constexpr,
        IS_DIVISIBLE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        EXPANDED_BIAS: tl.constexpr,
        PERF_OPT: tl.constexpr,
        IS_Y: tl.constexpr):
    # range of values handled by this stage
    if CAUSAL:
        lo, hi = 0, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            #k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16)
        if HAS_BIAS:
            bias = None
            if EXPANDED_BIAS:
                if IS_DIVISIBLE:
                    bias = tl.load(BIAS + off_z * N_CTX * N_CTX + offs_m[:, None] * N_CTX + start_n + offs_n)
                else:
                    bias = tl.load(BIAS + off_z * N_CTX * N_CTX + offs_m[:, None] * N_CTX + start_n + offs_n, mask=(offs_m < N_CTX)[:, None] & (offs_n < N_CTX)[None, :], other=0.)
            else:
                if IS_DIVISIBLE:
                    bias = tl.load(BIAS + off_z * N_CTX + start_n + offs_n)[None, :]
                else:
                    bias = tl.load(BIAS + off_z * N_CTX + start_n + offs_n, offs_n < N_CTX, other=0.)[None, :]
            qk += tl.dot(q, k)
            qk = qk * qk_scale
            #qk = qk * 0.0002450379808870375
            qk = qk + bias
        else:
            qk += tl.dot(q, k)
            qk = qk * qk_scale
            #qk = qk * 0.0002450379808870375
        if CAUSAL and start_n >= start_m * BLOCK_M:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
        qk = qk.to(tl.bfloat16)
        p = silu_forward(qk, PERF_OPT, IS_Y)
        if IS_DIVISIBLE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            #v = tl.load(V_block_ptr)
        qk_wram = p.to(v.type.element_ty)
        qkv = tl.dot(qk_wram, v)
        # -- update output accumulator --
        acc += qkv
  
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
  
    return acc
  
  
# We don't run auto-tuning everytime to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
#@triton.autotune(
#   configs=[
#       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=1),
#       #triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=1),
#       #triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=1),
#       #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=1),
#       #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=1),
#       #triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=1),
#   ],
#   key=['N_CTX'],
#)
  
  
@triton.jit
def _attn_fwd(
        Q,
        K,
        V,
        sm_scale,
        BIAS,
        Out,  #
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        Z,
        H,  #
        N_CTX: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_DMODEL: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        CAUSAL: tl.constexpr,  #
        IS_DIVISIBLE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        EXPANDED_BIAS: tl.constexpr,
        PERF_OPT: tl.constexpr,
        IS_Y: tl.constexpr):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    context_num = triton.cdiv(N_CTX, BLOCK_M)
    total_heads = Z * H * context_num
    deal_num_per_core = total_heads // program_dim
    extra_deal_core_num = total_heads - deal_num_per_core * program_dim
    deal_num_per_core += 1
    core_head_begin = program_id * deal_num_per_core
    if program_id >= extra_deal_core_num:
        deal_num_per_core -= 1
        core_head_begin = program_id * deal_num_per_core + extra_deal_core_num
    if deal_num_per_core <= 0:
        return
    head_begin = core_head_begin
    head_end = head_begin + deal_num_per_core
  
    for head_idx in range(head_begin, head_end):
        start_m = head_idx % context_num
        off_hz = head_idx // context_num
        off_z = off_hz // H
        off_h = off_hz % H
        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
        #import pdb
        #pdb.set_trace()
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + o_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        #qk_scale *= 1.44269504  # 1/log(2)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            #q = tl.load(Q_block_ptr)
        acc = _attn_fwd_inner(
            off_z,
            acc,
            q,
            BIAS,
            K_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,  #
            CAUSAL,
            offs_m,
            offs_n,
            N_CTX,  #
            IS_DIVISIBLE,
            HAS_BIAS,
            EXPANDED_BIAS,
            PERF_OPT,
            IS_Y)
  
        if IS_DIVISIBLE:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))
  
  
empty = torch.empty(128, device="mlu")
  
  
class _attention(torch.autograd.Function):
  
    @staticmethod
    def forward(ctx, q, k, v, bias, causal, sm_scale, has_bias, expanded_bias, perf_opt):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        o = torch.empty_like(q)
        if IS_Y:
            BLOCK_M = 256
            BLOCK_N = 128
            if perf_opt == 2:
                BLOCK_M = 128
            if Lq == 128 and perf_opt == 1:
                BLOCK_M = 128
            if Lq == 256:
                BLOCK_M = 128
                if perf_opt == 1 or perf_opt == 2:
                    BLOCK_N = 64
        else:
            BLOCK_M = 128
            BLOCK_N = 128
            if (Lq == 128 or Lq == 96) and perf_opt == 1:
                BLOCK_N = 64
            if Lq == 256:
                BLOCK_N = 64
                if perf_opt == 1:
                    BLOCK_M = 64
  
        num_warps = 1
        num_stages = 3
        processor_count = torch.mlu.get_device_properties(
            torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)
  
        def is_divisible(a, b):
            if b == 0:
                raise ValueError("Divisor cannot be 0")
            return a % b == 0
  
        N_CTX = q.shape[2]
        IS_DIVISIBLE = False
        if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
            IS_DIVISIBLE = True
        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            bias,
            o,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            q.shape[0],
            q.shape[1],  #
            N_CTX,  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            BLOCK_DMODEL=Lk,  #
            CAUSAL=causal,  #
            IS_DIVISIBLE=IS_DIVISIBLE,  #
            HAS_BIAS=has_bias,
            EXPANDED_BIAS=expanded_bias,
            PERF_OPT=perf_opt,
            IS_Y=IS_Y,
            num_warps=num_warps,  #
            num_stages=num_stages  #
        )
  
        return o
  
  
attention = _attention.apply
  
  
def naive(q, k, v, bias, causal, sm_scale, has_bias):
    N_CTX = q.shape[-2]
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="mlu"))
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    p = torch.matmul(q, k.transpose(2, 3)).float() * sm_scale
    if has_bias:
        bias = bias.unsqueeze(1).repeat(1, q.shape[1], 1, 1)
        p = p.masked_fill(bias == 0, 0.)
    if causal:
        p[:, :, M == 0] = float("-1e6")
    #p = (torch.nn.functional.silu(p)).bfloat16()
    p = (torch.nn.functional.silu(p) * 0.0002450379808870375).bfloat16()
    #p = p.bfloat16()
    ref_out = torch.matmul(p, v)
    return ref_out
  

@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(2, 8, 4081, 96)])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("has_bias", [False])
# bias supports 2d or 4d layout
@pytest.mark.parametrize("expanded_bias", [True])
@pytest.mark.parametrize("perf_opt", [0])
def test_op(Z, H, N_CTX, D_HEAD, causal, has_bias, expanded_bias, perf_opt, dtype=torch.bfloat16):
    torch.manual_seed(20)
    #q = torch.load('../qkv_input/q.pt').contiguous()
    #k = torch.load('../qkv_input/k.pt').contiguous()
    #v = torch.load('../qkv_input/v.pt').contiguous()
    #q = torch.load('../qkv_input/q.pt')
    #k = torch.load('../qkv_input/k.pt')
    #v = torch.load('../qkv_input/v.pt')
    q = torch.load('../qkv_input/q.pt')
    k = torch.load('../qkv_input/k.pt')
    v = torch.load('../qkv_input/v.pt')
    if isinstance(q, torch.Tensor):
        print("张量的形状:", q.dtype)
        print("张量的形状:", q.shape)
        print("张量的形状num:", q.numel())
        print("张量stride:", q.stride())
        print("张量的形状:", k.shape)
        print("张量的形状:", k.stride())
        print("张量的形状:", v.shape)
        print("张量的形状:", v.stride())

    #q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
    #                 device="mlu").normal_(mean=0.0, std=0.1).requires_grad_())
    #k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
    #                 device="mlu").normal_(mean=0.0, std=0.1).requires_grad_())
    #v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype,
    #                 device="mlu").normal_(mean=0.0, std=0.1).requires_grad_())
    #q = to_triton(torch.load('../qkv_input/q.pt'), device='mlu')
    #k = to_triton(torch.load('../qkv_input/k.pt'), device='mlu')
    #v = to_triton(torch.load('../qkv_input/v.pt'), device='mlu')
    
    #q = torch.tensor(torch.load('../qkv_input/q.pt'), device='mlu')
    #k = torch.tensor(torch.load('../qkv_input/k.pt'), device='mlu')
    #v = torch.tensor(torch.load('../qkv_input/v.pt'), device='mlu')

    q = q.to("mlu")
    k = k.to("mlu")
    v = v.to("mlu")

    if isinstance(q, torch.Tensor):
        print("张量的形状:", q.shape)
        print("张量的形状num:", q.numel())
        print("张量stride:", q.stride())
        print("张量的形状:", k.shape)
        print("张量的形状:", k.stride())
        print("张量的形状:", v.shape)
        print("张量的形状:", v.stride())

    naive_bias = None
    bias = None
    if has_bias:
        naive_bias = torch.rand(size=(Z, N_CTX, N_CTX),
                                dtype=torch.float32,
                                device='mlu')
        if expanded_bias:
            naive_bias = torch.bernoulli(naive_bias)
            bias = naive_bias
        else:
            bias = torch.rand(size=(Z, N_CTX),
                                    dtype=torch.float32,
                                    device='mlu')
            bias = torch.bernoulli(bias)
            naive_bias = bias.repeat(1, q.shape[2], 1)
    sm_scale = 1.0
    #sm_scale = 0.0002450379808870375
    # reference implementation
    ref_out = naive(q, k, v, naive_bias, causal, sm_scale, has_bias)
    # triton implementation
    # use add instead of mul, fix bias value
    if has_bias:
        bias = torch.where(bias == 1, 0.0, -1e8)
    tri_out = attention(q, k, v, bias, causal, sm_scale, has_bias, expanded_bias, perf_opt).bfloat16()
    print(tri_out[0, 7, 2382, 4])
    print(ref_out[0, 7, 2382, 4])
    print(tri_out[0, 3, 849, 12])
    print(ref_out[0, 3, 849, 12])
    ## compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-3)
  
  
BATCH, N_HEADS, N_CTX, D_HEAD = 32, 8, 1024, 128
# vary seq length for fixed head and batch=4
configs = []
for causal in [True]:
    for has_bias in [True]:
        for expanded_bias in [True]:
            for perf_opt in [0, 1, 2]:
                for D_HEAD in [64, 128, 256]:
                    configs.append(
                        triton.testing.Benchmark(
                            x_names=["N_CTX"],
                            #x_vals=[2**14],  #10-14,1024~...
                            x_vals=[2**i for i in range(10, 12)],  #10-14,1024~...
                            #x_vals=[2**i for i in range(11, 12)],  #10-14,1024~...
                            line_arg="provider",
                            line_vals=["triton"] + ["naive"],
                            line_names=["Triton"] + ["Naive"],
                            styles=[("red", "-"), ("blue", "-")],
                            ylabel="ms",
                            plot_name=
                            f"linear-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-causal={causal}-has_bias={has_bias}-perf_opt={perf_opt}",
                            args={
                                "H": N_HEADS,
                                "BATCH": BATCH,
                                "D_HEAD": D_HEAD,
                                "dtype": torch.half,
                                "causal": causal,
                                "has_bias": has_bias,
                                "expanded_bias": expanded_bias,
                                "perf_opt": perf_opt,
                            },
                        ))
  
  
@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH,
                          H,
                          N_CTX,
                          D_HEAD,
                          causal,
                          has_bias,
                          expanded_bias,
                          perf_opt,
                          provider,
                          dtype=torch.half,
                          device="mlu"):
    warmup = 5
    rep = 8
    q = torch.randn((BATCH, H, N_CTX, D_HEAD),
                    dtype=dtype,
                    device="mlu",
                    requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD),
                    dtype=dtype,
                    device="mlu",
                    requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD),
                    dtype=dtype,
                    device="mlu",
                    requires_grad=True)
    bias = None
    naive_bias = None
    if has_bias:
        naive_bias = torch.randn((BATCH, N_CTX, N_CTX),
                          dtype=torch.float32,
                          device="mlu",
                          requires_grad=True)
        if expanded_bias:
            bias = naive_bias
        else:
            bias = torch.rand(size=(BATCH, N_CTX),
                                    dtype=torch.float32,
                                    device='mlu')
    sm_scale = 1.3
  
    if provider == "triton":
        fn = lambda: attention(q, k, v, bias, causal, sm_scale, has_bias, expanded_bias, perf_opt)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "naive":
        fn = lambda: naive(q, k, v, naive_bias, causal, sm_scale, has_bias)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms
  
  
if __name__ == "__main__":
    bench_flash_attention.run(print_data=True)
