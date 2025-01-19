import torch
import triton.language as tl
import triton
import ipdb

triton.Config.debug = True

def mem_update(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lr: torch.Tensor, w: torch.Tensor, wts: torch.Tensor, B: int, T: int, H: int, d: int):
    for t in range(T):
        wts[:, :, t] = w
        o[:, t] = (q[:, t].view(B, H, 1, d) @ w).view(B, H, d) # recall first
        dw = k[:, t].view(B, H, d, 1) @ (v[:, t].view(B, H, 1, d) - k[:, t].view(B, H, 1, d) @ w)
        dw = lr[:, t].view(B, H, 1, d) * dw
        w = w + dw

class MemUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, lr, B, T, H, d):
        o = torch.empty_like(q)
        w = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        wts = torch.empty(B, H, T, d, d, device=q.device, dtype=q.dtype)
        mem_update(q, k, v, o, lr, w, wts, B, T, H, d)
        ctx.save_for_backward(q, k, v, o, lr, wts)
        ctx.B, ctx.T, ctx.H, ctx.d = B, T, H, d
        return o, w, wts
    
    @staticmethod
    def backward(ctx, o_grad, w_grad, wts_grad):
        # w_grad and wts_grad are None
        q, k, v, o, lr, wts = ctx.saved_tensors
        B, T, H, d = ctx.B, ctx.T, ctx.H, ctx.d
        q_grad = torch.empty_like(q)
        k_grad = torch.empty_like(k)
        v_grad = torch.empty_like(v)
        lr_grad = torch.empty_like(lr)

        # at the last step, k, v and lr actually never contribute to the result. only q has its grad
        q_grad[:, -1] = (o_grad[:, -1].view(B, H, 1, d) @ wts[:, :, -1].transpose(-1, -2)).view(B, H, d)
        k_grad[:, -1] = torch.zeros_like(k[:, -1])
        v_grad[:, -1] = torch.zeros_like(v[:, -1])
        lr_grad[:, -1] = torch.zeros_like(lr[:, -1])

        w_grad_i = q[:, -1].view(B, H, d, 1) @ o_grad[:, -1].view(B, H, 1, d)

        for t in range(T-2, -1, -1):
            dw = wts[:, :, t+1] - wts[:, :, t]
            lr_grad[:, t] = (w_grad_i * dw).sum(dim=0)
            q_grad[:, t] = (o_grad[:, t].view(B, H, 1, d) @ wts[:, :, t].transpose(-1, -2)).view(B, H, d)
            scaled_w_grad = lr[:, t].view(B, H, 1, d) * w_grad_i
            v_grad[:, t] = (k[:, t].view(B, H, 1, d) @ scaled_w_grad).view(B, H, d)
            w_scaled_w_grad = wts[:, :, t] @ scaled_w_grad.transpose(-1, -2)
            k_grad[:, t] = (v[:, t].view(B, H, 1, d) @ scaled_w_grad.transpose(-1, -2)).view(B, H, d)
            k_grad[:, t] -= (k[:, t].view(B, H, 1, d) @ (w_scaled_w_grad - w_scaled_w_grad.transpose(-1, -2))).view(B, H, d)
            w_grad_i += q[:, t].view(B, H, d, 1) @ o[:, t].view(B, H, 1, d)
            w_grad_i -= k[:, t].view(B, H, d, 1) @ (k[:, t].view(B, H, 1, d) @ scaled_w_grad)

        return (q_grad, k_grad, v_grad, lr_grad, None, None, None, None)

def mem_update_hand(q, k, v, lr, B, T, H, d):
    return MemUpdate.apply(q, k, v, lr, B, T, H, d)

@triton.jit
def vec_mat_mul(vec, mat):
    """
    vec: [A]
    mat: [B, A]
    """
    return tl.sum(vec[None, :] * mat, axis=1)

@triton.jit
def mem_update_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lr_ptr, w_ptr, wts_ptr, 
                          B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, d: tl.constexpr, 
                          CHUNK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    q, k, v, o, lr: [B, H, T, d]
    w: [B, H, d, d]^T
    wts: [B, H, T, d, d]^T

    launch with grid: [B, H, ceil(d // BLOCK_SIZE)]
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    d_col_block_id = tl.program_id(2)

    stride_b = H * T * d
    stride_h = T * d
    stride_t = d 

    stride_w_b = H * d * d
    stride_w_h = d * d
    stride_w_d = d

    stride_wts_b = H * T * d * d
    stride_wts_h = T * d * d
    stride_wts_t = d * d
    stride_wts_d = d

    q_base_ptr = q_ptr + batch_id * stride_b + head_id * stride_h + 0 * stride_t + tl.arange(0, d)
    k_base_ptr = k_ptr + batch_id * stride_b + head_id * stride_h + 0 * stride_t + tl.arange(0, d)
    lr_base_ptr = lr_ptr + batch_id * stride_b + head_id * stride_h + 0 * stride_t + d_col_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    v_base_ptr = v_ptr + batch_id * stride_b + head_id * stride_h + 0 * stride_t + d_col_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    o_base_ptr = o_ptr + batch_id * stride_b + head_id * stride_h + 0 * stride_t + d_col_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    w_base_ptr = w_ptr + batch_id * stride_w_b + head_id * stride_w_h + d_col_block_id * BLOCK_SIZE * stride_w_d + 0
    wts_base_ptr = wts_ptr + batch_id * stride_wts_b + head_id * stride_wts_h + 0 * stride_wts_t + d_col_block_id * BLOCK_SIZE * stride_wts_d + 0

    w = tl.load(w_base_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_w_d + tl.arange(0, d)[None, :])

    for i in range(T):
        q = tl.load(q_base_ptr)
        k = tl.load(k_base_ptr)
        v = tl.load(v_base_ptr)
        lr = tl.load(lr_base_ptr)
        tl.store(wts_base_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_wts_d + tl.arange(0, d)[None, :], w)
        o = tl.sum(q[None, :] * w, axis=1)
        diff = v - tl.sum(k[None, :] * w, axis=1)
        dw = k[:, None] * diff[None, :]
        dw = lr[None, :] * dw
        w += dw.trans()   # TODO: maybe less efficient
        tl.store(o_base_ptr, o)

        q_base_ptr += stride_t
        k_base_ptr += stride_t
        v_base_ptr += stride_t
        lr_base_ptr += stride_t
        o_base_ptr += stride_t
        wts_base_ptr += stride_wts_t

    tl.store(w_base_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_w_d + tl.arange(0, d)[None, :], w)

# def mem_update_bwd_kernel(x_ptr, )

def mem_update_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lr: torch.Tensor, B: int, T: int, H: int, d: int):

    B, T, H, d = q.shape

    # [B, T, H, d] -> [B, H, T, d]
    o = torch.empty_like(q)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()
    lr = lr.transpose(1, 2).contiguous()
    w = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)

    # save for backward
    wts = torch.empty(B, H, T, d, d, device=w.device, dtype=w.dtype)

    assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda and lr.is_cuda and w.is_cuda
    grid = lambda meta: (B, H, triton.cdiv(d, meta['BLOCK_SIZE']))
    mem_update_fwd_kernel[grid](q, k, v, o, lr, w, wts, B, T, H, d, CHUNK_SIZE=32, BLOCK_SIZE=32)
    torch.cuda.synchronize()

    # [B, H, T, d] -> [B, T, H, d]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()
    lr = lr.transpose(1, 2).contiguous()
    w = w.transpose(-1, -2)

    # print(o.flatten(-2, -1).shape)
    # print(o.flatten(-2, -1)[0])

    # print(o[0, :, 0, :])
    return o, w, wts