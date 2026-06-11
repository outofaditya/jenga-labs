import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    B,
    H,
    N_CTX,
    Out,
    # strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kk,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    HEAD_DIM: tl.constexpr,  # 128
):
    # per-head relu(qk^t) summed then block-max into Out; caller divides by 64
    row_block_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    row_start = row_block_idx * BLOCK_M
    col_start = col_block_idx * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)  # 0..63
    offs_n = tl.arange(0, BLOCK_N)  # 0..63
    offs_dim = tl.arange(0, HEAD_DIM)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for h_idx in range(0, H):
        q_ptrs = (
            Q
            + b_idx * stride_qb
            + h_idx * stride_qh
            + (row_start + offs_m[:, None]) * stride_qm
            + offs_dim[None, :] * stride_qk
        )
        q_block = tl.load(q_ptrs)  # [BLOCK_M, HEAD_DIM]

        k_ptrs = (
            K
            + b_idx * stride_kb
            + h_idx * stride_kh
            + offs_dim[:, None] * stride_km
            + (col_start + offs_n[None, :]) * stride_kk
        )
        k_block = tl.load(k_ptrs)

        partial = tl.dot(q_block, k_block)  # float32

        partial = tl.maximum(partial, 0.0)
        acc += partial

    row_max = tl.max(acc, axis=1)
    block_max = tl.max(row_max, axis=0)

    out_ptrs = (
        Out + b_idx * stride_ob + row_block_idx * stride_om + col_block_idx * stride_on
    )
    tl.store(out_ptrs, block_max.to(Out.type.element_ty))


def block_attn_pool(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # k must arrive pre-transposed to [B, H, D, N]; output is [B, N//64, N//64]
    assert q.is_cuda and k.is_cuda, "Triton kernel only works on GPU tensors."
    B, H, N, Dq = q.shape
    Bk, Hk, Dk, Nk = k.shape
    assert B == Bk and H == Hk and N == Nk and Dq == Dk, "q, k shape mismatch."
    BLOCK = 64
    assert N % BLOCK == 0, "For simplicity, assume N is multiple of 64."

    out = torch.empty(
        (B, N // BLOCK, N // BLOCK), dtype=torch.bfloat16, device=q.device
    )

    # Q: shape [B, H, N, D]
    stride_qb = q.stride(0)
    stride_qh = q.stride(1)
    stride_qm = q.stride(2)
    stride_qk = q.stride(3)
    # K: shape [B, H, D, N]
    stride_kb = k.stride(0)
    stride_kh = k.stride(1)
    stride_km = k.stride(2)
    stride_kk = k.stride(3)
    # Out: shape [B, N//64, N//64]
    stride_ob = out.stride(0)
    stride_om = out.stride(1)
    stride_on = out.stride(2)

    grid = (N // BLOCK, N // BLOCK, B)

    _fwd_kernel[grid](
        q,
        k,
        B,
        H,
        N,
        out,
        # strides
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_km,
        stride_kk,
        stride_ob,
        stride_om,
        stride_on,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        HEAD_DIM=128,
        num_stages=2,
        num_warps=4,
    )

    out /= 64.0
    return out


def demo():
    # sanity check: random q, k, compare triton output against the matmul reference
    B, H, N, D = 1, 32, 1024, 128
    device = "cuda"

    q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device=device)
    k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device=device)

    out_triton = block_attn_pool(q, k.transpose(2, 3).contiguous())

    attn_weight = torch.matmul(q, k.transpose(-2, -1))
    for i in range(attn_weight.size(1)):
        attn_weight[:, i][attn_weight[:, i] < 0] = 0
    attn_weight = attn_weight.sum(dim=1)
    attn_maxpool = torch.nn.functional.max_pool2d(
        attn_weight, kernel_size=64, stride=64
    )  # [B, N//64, N//64]
    attn_maxpool /= 64

    diff = (out_triton - attn_maxpool).abs().max().item()
    print(f"max abs diff = {diff:.6f}")
    if diff < 1e-2:
        print("Test PASS!")
    else:
        print("Test FAIL!")
    print(out_triton)
    print(attn_maxpool)


if __name__ == "__main__":
    demo()
