import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_lower_triangle(
    Q,
    K,
    B,
    H,
    N_CTX,
    Out,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INV_SQRT_HEAD_DIM: tl.constexpr,
):
    # lower-triangular relu(qk^t)/sqrt(d) summed over heads then max-pooled per 64x64 block
    row_block_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    row_start = row_block_idx * BLOCK_M
    col_start = col_block_idx * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for h_idx in range(0, H):
        q_ptrs = (
            Q
            + b_idx * stride_qb
            + h_idx * stride_qh
            + (row_start + offs_m[:, None]) * stride_qm
            + offs_d[None, :] * stride_qk
        )
        q_block = tl.load(q_ptrs)

        k_ptrs = (
            K
            + b_idx * stride_kb
            + h_idx * stride_kh
            + offs_d[:, None] * stride_km
            + (col_start + offs_n[None, :]) * stride_kk
        )
        k_block = tl.load(k_ptrs)

        partial = tl.dot(q_block, k_block)
        partial = partial * INV_SQRT_HEAD_DIM

        cond = (row_start + offs_m[:, None]) >= (col_start + offs_n[None, :])
        partial = tl.where(cond, partial, 0.0)
        partial = tl.maximum(partial, 0.0)

        acc += partial

    row_max = tl.max(acc, axis=1)
    block_max = tl.max(row_max, axis=0)

    out_ptrs = (
        Out + b_idx * stride_ob + row_block_idx * stride_om + col_block_idx * stride_on
    )
    tl.store(out_ptrs, block_max.to(Out.type.element_ty))


def block_lower_triangle_attn_pool(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # k must arrive pre-transposed to [B, H, D, N]; output is [B, N//64, N//64]
    assert q.is_cuda and k.is_cuda, "Triton kernel only works on GPU."
    B, H, N, Dq = q.shape
    Bk, Hk, Dk, Nk = k.shape
    assert B == Bk and H == Hk and N == Nk and Dq == Dk, "q, k shape mismatch."
    BLOCK = 64
    assert N % BLOCK == 0, "For simplicity, assume N is multiple of 64."

    out = torch.empty((B, N // BLOCK, N // BLOCK), dtype=torch.float32, device=q.device)

    stride_qb, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kb, stride_kh, stride_km, stride_kk = k.stride()
    stride_ob, stride_om, stride_on = out.stride()

    inv_sqrt_d = float(1.0 / math.sqrt(Dq))

    grid = (N // BLOCK, N // BLOCK, B)

    _fwd_kernel_lower_triangle[grid](
        q,
        k,
        B,
        H,
        N,
        out,
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
        HEAD_DIM=Dq,
        INV_SQRT_HEAD_DIM=inv_sqrt_d,
        num_stages=2,
        num_warps=4,
    )

    return out


class PrunedLlamaMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        gate_w,
        gate_b,
        up_w,
        up_b,
        down_w,
        down_b,
        prune_ratio,
        layer_idx,
        show_mlp=False,
    ):
        # frozen gate/up/down weights; gradient flows only through x via the kept columns
        B, S, H = x.shape
        I = gate_w.shape[0]

        gate = torch.matmul(x, gate_w.t())
        up = torch.matmul(x, up_w.t())

        a = F.silu(gate)
        gate_2d = a.reshape(-1, I)
        up_2d = up.reshape(-1, I)
        col_norm_gate = gate_2d.norm(p=2, dim=0)
        col_norm_up = up_2d.norm(p=2, dim=0)

        col_norm_sum = col_norm_gate + col_norm_up
        keep_num = int(I * (1 - prune_ratio))

        top_values, top_indices = torch.topk(col_norm_sum, keep_num, largest=True)
        min_value = top_values[-1]
        max_value = top_values[0]
        if show_mlp:
            print(
                f"layer: {layer_idx} , threshold: {min_value / max_value:.4f}, memory: {1 - prune_ratio}"
            )
        gate_small = torch.index_select(gate, dim=2, index=top_indices)
        up_small = torch.index_select(up, dim=2, index=top_indices)
        a_small = torch.index_select(a, dim=2, index=top_indices)

        c = a * up
        out = torch.matmul(c, down_w.t())
        ctx.save_for_backward(
            gate_small,
            up_small,
            a_small,
            top_indices,
            gate_w,
            up_w,
            down_w,
        )
        ctx.prune_ratio = prune_ratio
        ctx.B = x.shape[0]
        ctx.S = x.shape[1]
        ctx.H = x.shape[2]
        ctx.I = I
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # weights are frozen so only dL/dx is returned; other slots are None
        (
            gate_small,
            up_small,
            a_small,
            top_indices,
            gate_w,
            up_w,
            down_w,
        ) = ctx.saved_tensors

        B, S, I = ctx.B, ctx.S, ctx.I

        grad_c = torch.matmul(grad_out, down_w)
        grad_c_small = torch.index_select(grad_c, dim=2, index=top_indices)

        grad_a_small = grad_c_small * up_small
        grad_up_small = grad_c_small * a_small

        # autograd is cheaper than hand-coding silu' here; gate_small is small after top-k
        with torch.enable_grad():
            gate_small_ = gate_small.detach().clone().requires_grad_(True)
            a_small_ = F.silu(gate_small_)
            grad_gate_small_ = torch.autograd.grad(
                a_small_.sum(), gate_small_, retain_graph=False
            )[0]
        grad_gate_small = grad_a_small * grad_gate_small_

        grad_gate = torch.zeros(
            [B, S, I], dtype=grad_gate_small.dtype, device=grad_gate_small.device
        )
        grad_up = torch.zeros(
            [B, S, I], dtype=grad_gate_small.dtype, device=grad_gate_small.device
        )
        grad_gate.index_copy_(2, top_indices, grad_gate_small)
        grad_up.index_copy_(2, top_indices, grad_up_small)

        grad_x_from_gate = torch.matmul(grad_gate, gate_w)
        grad_x_from_up = torch.matmul(grad_up, up_w)
        grad_x = grad_x_from_gate + grad_x_from_up

        return (grad_x, None, None, None, None, None, None, None, None, None)
