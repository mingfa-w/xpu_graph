import torch
import torch_mlu

import triton
import triton.language as tl

import pytest

test_case_M_K1_N1_N2 = [
    (160, 128, 64, 32),
    (384, 128, 64, 32),
    (512, 128, 64, 32),
]


def do_config_prune(configs, named_args, **kwargs):
    M = named_args["M"]
    block_set = set()
    if M < 512:
        block_set.add(M)

    for config in configs:
        block_size_m = config.kwargs["BLOCK_SIZE_M"]
        block_set.add(block_size_m)

    assert block_set, "no valid configs available"

    pruned_configs = []
    for block_size_m in block_set:
        pruned_configs.append(
            triton.Config({
                "BLOCK_SIZE_M": block_size_m,
            },
                          num_stages=1,
                          num_warps=1))

    return pruned_configs


configs = [
    triton.Config({'BLOCK_SIZE_M': BM}, num_stages=1, num_warps=1)
    for BM in [128, 256, 512]
]

@triton.jit
def relu(x):
    return tl.maximum(x, 0.0)

@triton.autotune(
    configs=configs,
    prune_configs_by={"early_config_prune": do_config_prune},
    key=['M', 'K1', 'N1', "N2"],
)
@triton.heuristics({
    'EVEN_M':
    lambda args: args['M'] % args['BLOCK_SIZE_M'] == 0,
})

@triton.jit
def fused_tinyffn_kernel(
    d_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    bias1_ptr,
    bias2_ptr,
    M,
    K1: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ck: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    is_up_bias: tl.constexpr,
    is_down_bias: tl.constexpr,
    is_up_act: tl.constexpr,
    is_down_act: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    offs_k1 = tl.arange(0, K1)
    offs_n1 = tl.arange(0, N1)
    offs_n2 = tl.arange(0, N2)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am +
                      offs_k1[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k1[:, None] * stride_bk +
                      offs_n1[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_n1[:, None] * stride_ck +
                      offs_n2[None, :] * stride_cn)

    if EVEN_M:
        a = tl.load(a_ptrs, cache_modifier=".ca")
    else:
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None],
                    other=0.0,
                    cache_modifier=".ca")
    b = tl.load(b_ptrs, cache_modifier=".ca")
    c = tl.load(c_ptrs, cache_modifier=".ca")

    if is_up_bias:
        bias1_ptrs = bias1_ptr + offs_n1
        bias1 = tl.load(bias1_ptrs, cache_modifier=".ca")
    if is_down_bias:
        bias2_ptrs = bias2_ptr + offs_n2
        bias2 = tl.load(bias2_ptrs, cache_modifier=".ca")

    d = tl.dot(a, b, allow_tf32=False)
    if is_up_bias:
        d = d + bias1
    if is_up_act:
        d = relu(d)

    d = tl.dot(d.to(d_ptr.type.element_ty), c, allow_tf32=False)
    if is_down_bias:
        d = d + bias2
    if is_down_act:
        d = relu(d)

    d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n2[None, :]
    if EVEN_M:
        tl.store(d_ptrs, d)
    else:
        tl.store(d_ptrs, d, mask=mask_m[:, None])

@torch.library.custom_op(
    "torch_mlu_triton::fuse_tinyffn", mutates_args=(), device_types="mlu"
)
def fuse_tinyffn(
    a: torch.Tensor,
    b: torch.Tensor,
    bias1: torch.Tensor,
    c: torch.Tensor,
    bias2: torch.Tensor,
    is_up_bias: bool,
    is_down_bias: bool,
    is_up_act: bool,
    is_down_act: bool,
) -> torch.Tensor:
    d = torch.empty(
        [a.shape[0], c.shape[1]],
        dtype=a.dtype,
        device=a.device,
    )

    M, K1 = a.shape
    K1, N1 = b.shape
    N1, N2 = c.shape

    # TODO: ASSERT K1/N1/N2
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), )
    fused_tinyffn_kernel[grid](d, a, b, c, bias1, bias2, M, K1, N1, N2, a.stride(0),
                              a.stride(1), b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1), d.stride(0),
                              d.stride(1), is_up_bias, is_down_bias,
                              is_up_act, is_down_act)
    return d

@fuse_tinyffn.register_fake
def fuse_tinyffn_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    bias1: torch.Tensor,
    c: torch.Tensor,
    bias2: torch.Tensor,
    is_up_bias: bool,
    is_down_bias: bool,
    is_up_act: bool,
    is_down_act: bool,
) -> torch.Tensor:
    # Check constraints.
    assert a.shape[1] == b.shape[0], "a, b Incompatible dimensions"
    assert b.shape[1] == c.shape[0], "b, c Incompatible dimensions"

    assert a.is_contiguous(), "a should be contiguous"
    assert b.is_contiguous(), "b should be contiguous"
    assert c.is_contiguous(), "c should be contiguous"
    #assert d.is_contiguous(), "d should be contiguous"

    assert a.dtype == b.dtype, "a, b should have the same dtype"
    assert b.dtype == c.dtype, "b, c should have the same dtype"

    M, K1 = a.shape
    K1, N1 = b.shape
    N1, N2 = c.shape
    #assert K1 <= 256 and N1 <= 256 and N2 <= 256, "K1, N1, and N2 should be <= 256"
    size_of_dtype = 2
    if a.dtype == torch.float32:
        size_of_dtype = 4
    assert K1 * N1 + N1 * N2 + N1 + N2 <= 512 / size_of_dtype * 1024, \
        "(K1 * N1 + N1 * N2 + N1 + N2) should be <= (512 / size_of_dtype * 1024)"

    d = torch.empty(
        [a.shape[0], c.shape[1]],
        dtype=a.dtype,
        device=a.device,
    )
    return d


#@pytest.mark.parametrize("M, K1, N1, N2", test_case_M_K1_N1_N2)
#def test_cast_gating(M, K1, N1, N2):
#    a_ = torch.randint(-5, 5, (M, K1), device="mlu")
#    a = a_.bfloat16()
#    b_ = torch.randint(-5, 5, (K1, N1), device="mlu")
#    b = b_.bfloat16()
#    c_ = torch.randint(-5, 5, (N1, N2), device="mlu")
#    c = c_.bfloat16()
#
#    d = torch.empty((M, N2), device=a.device, dtype=torch.bfloat16)
#
#    triton_output = fused_matmul(a, b, c, d)
#
#    torch_output = torch.mm(torch.mm(a, b), c)
#
#    torch.testing.assert_close(triton_output, torch_output, atol=0, rtol=0)
#
#
#@triton.testing.perf_report(
#    triton.testing.Benchmark(
#        x_names=["M", "K1", "N1",
#                 "N2"],  # Argument names to use as an x-axis for the plot
#        x_vals=test_case_M_K1_N1_N2,
#        line_arg='provider',
#        line_vals=['torch', 'triton'],
#        line_names=["Torch (ms)", "Triton (ms)"],
#        styles=[('blue', '-'), ("red", "-")],
#        ylabel="Time",
#        plot_name="fused_matmul-performance",
#        args={},
#    ))
#def benchmark(M, K1, N1, N2, provider):
#    dtype = torch.bfloat16
#    device = "mlu"
#    a = torch.randn((M, K1), device=device, dtype=dtype)
#    b = torch.randn((K1, N1), device=device, dtype=dtype)
#    c = torch.randn((N1, N2), device=device, dtype=dtype)
#
#    if provider == "torch":
#        fn = lambda: torch.mm(torch.mm(a, b), c)
#
#    if provider == 'triton':
#        d = torch.empty((M, N2), device=device, dtype=dtype)
#        fn = lambda: fused_matmul(a, b, c, d)
#
#    ms_ = triton.testing.do_bench(fn)
#    ms = triton.testing.do_bench(fn, warmup=10 * ms_, rep=102 * ms_)
#
#    return ms
#
#
#if __name__ == "__main__":
#    benchmark.run(show_plots=True, print_data=True)
