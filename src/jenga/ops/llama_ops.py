import torch
import torch.nn.functional as F


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
        prune_ratio=0.5,
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

        _, top_indices = torch.topk(col_norm_sum, keep_num, largest=True)
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

        return (grad_x, None, None, None, None, None, None, None)
