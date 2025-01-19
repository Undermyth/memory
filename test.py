import torch
import torch.nn as nn
from transformers import PretrainedConfig
import torch.nn.functional as F
from ops import mem_update, mem_update_fwd, mem_update_hand
import time
import unittest

def detailed_allclose_check(ref: torch.Tensor, test: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5):
    # 检查形状是否一致
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: ref.shape = {ref.shape}, test.shape = {test.shape}")
    
    # 检查NaN，并将双方都是NaN的情况认为是相等
    both_nan_mask = torch.isnan(ref) & torch.isnan(test)
    
    # 替换NaN为零以避免对allclose的影响
    ref_no_nan = torch.where(both_nan_mask, torch.zeros_like(ref), ref)
    test_no_nan = torch.where(both_nan_mask, torch.zeros_like(test), test)
    
    # 使用allclose检查
    close_mask = torch.isclose(ref_no_nan, test_no_nan, atol=atol, rtol=rtol)
    close_mask |= both_nan_mask  # NaN对齐视为相等

    # 如果所有位置都符合预期，返回成功信息
    if close_mask.all():
        print("All values are within the tolerance.")
        return True
    
    # 找到第一个不符合的索引
    mismatched_indices = torch.nonzero(~close_mask, as_tuple=True)
    first_mismatch_idx = tuple(idx[0].item() for idx in mismatched_indices)
    
    # 输出详细错误信息
    ref_val = ref[first_mismatch_idx].item()
    test_val = test[first_mismatch_idx].item()
    raise AssertionError(
        f"Mismatch found at index {first_mismatch_idx}:\n"
        f"  Reference value: {ref_val}\n"
        f"  Test value: {test_val}\n"
        f"  Allowed tolerance: atol={atol}, rtol={rtol}\n"
        f"  Difference: {abs(ref_val - test_val)}, {abs(ref_val - test_val) / abs(ref_val)}\n"
        f"  Reference tensor contains NaN: {torch.isnan(ref).any().item()}\n"
        f"  Test tensor contains NaN: {torch.isnan(test).any().item()}"
    )


class Config(PretrainedConfig):
    def __init__(self, hidden_size, wm_head):
        super().__init__()
        self.hidden_size = hidden_size
        self.wm_head = wm_head

class WorkingMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.w_lr = nn.Linear(config.hidden_size, config.hidden_size)
        #nn.init.normal_(self.w_lr.weight, std=config.init_std)
        nn.init.constant_(self.w_lr.bias, 0)
        
        self.wm_head = config.wm_head
        self.wm_size = config.hidden_size // config.wm_head
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        B, T, D = q.size()
        H = self.wm_head
        d = self.wm_size

        q = q.view(B, T, self.wm_head, -1)
        k = k.view(B, T, self.wm_head, -1)
        v = v.view(B, T, self.wm_head, -1)
        
        #q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)
        
        lr = torch.sigmoid(self.w_lr(x)).view(B, T, self.wm_head, -1)# * self.lr
        
        o = torch.empty_like(q)
        w = torch.zeros((B, H, d, d), device=q.device, dtype=q.dtype)
        
        for t in range(T):
            o[:, t] = (q[:, t].view(B, H, 1, d) @ w).view(B, H, d) # recall first
            dw = k[:, t].view(B, H, d, 1) @ (v[:, t].view(B, H, 1, d) - k[:, t].view(B, H, 1, d) @ w)
            dw = lr[:, t].view(B, H, 1, d) * dw
            w = w + dw
            
        o = o.view(B, T, -1)
        o = self.out(o)
        
        #print(w.norm().item())
        
        return o, w     # NOTE: return w just for accuracy testing

class HandGradWorkingMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.w_lr = nn.Linear(config.hidden_size, config.hidden_size)
        #nn.init.normal_(self.w_lr.weight, std=config.init_std)
        nn.init.constant_(self.w_lr.bias, 0)
        
        self.wm_head = config.wm_head
        self.wm_size = config.hidden_size // config.wm_head
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        B, T, D = q.size()
        H = self.wm_head
        d = self.wm_size

        q = q.view(B, T, self.wm_head, -1)
        k = k.view(B, T, self.wm_head, -1)
        v = v.view(B, T, self.wm_head, -1)
        
        #q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)
        
        lr = torch.sigmoid(self.w_lr(x)).view(B, T, self.wm_head, -1)# * self.lr
        
        o = torch.empty_like(q)
        w = torch.zeros((B, H, d, d), device=q.device, dtype=q.dtype)
        
        # for t in range(T):
        #     o[:, t] = (q[:, t].view(B, H, 1, d) @ w).view(B, H, d) # recall first
        #     dw = k[:, t].view(B, H, d, 1) @ (v[:, t].view(B, H, 1, d) - k[:, t].view(B, H, 1, d) @ w)
        #     dw = lr[:, t].view(B, H, 1, d) * dw
        #     w = w + dw
        o, w, wts = mem_update_hand(q, k, v, lr, B, T, H, d)

        o = o.view(B, T, -1)
        o = self.out(o)
        
        #print(w.norm().item())
        
        return o, w     # NOTE: return w just for accuracy testing
    
class FastWorkingMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.w_lr = nn.Linear(config.hidden_size, config.hidden_size)
        #nn.init.normal_(self.w_lr.weight, std=config.init_std)
        nn.init.constant_(self.w_lr.bias, 0)
        
        self.wm_head = config.wm_head
        self.wm_size = config.hidden_size // config.wm_head
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        B, T, D = q.size()
        H = self.wm_head
        d = self.wm_size

        q = q.view(B, T, self.wm_head, -1)
        k = k.view(B, T, self.wm_head, -1)
        v = v.view(B, T, self.wm_head, -1)
        
        #q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)
        
        lr = torch.sigmoid(self.w_lr(x)).view(B, T, self.wm_head, -1)# * self.lr
        
        o = torch.empty_like(q)
        w = torch.zeros((B, H, d, d), device=q.device, dtype=q.dtype)
        
        # for t in range(T):
        #     o[:, t] = (q[:, t].view(B, H, 1, d) @ w).view(B, H, d) # recall first
        #     dw = k[:, t].view(B, H, d, 1) @ (v[:, t].view(B, H, 1, d) - k[:, t].view(B, H, 1, d) @ w)
        #     dw = lr[:, t].view(B, H, 1, d) * dw
        #     w = w + dw
        # NOTE: result should be explicitly assigned for synchronization
        o, w, wts = mem_update_fwd(q, k, v, lr, B, T, H, d)
        
        o = o.view(B, T, -1)
        o = self.out(o)
        
        return o, w     # NOTE: return w just for accuracy testing
    
class TestWorkingMemory(unittest.TestCase):

    def setUp(self):
        # 初始化配置和模型
        self.config = Config(256, 4)
        self.ref_model = WorkingMemory(self.config).cuda()
        self.test_model = FastWorkingMemory(self.config).cuda()
        # self.test_model = HandGradWorkingMemory(self.config).cuda()
        self.x = torch.randn(4, 1024, 256).cuda()

        # 确保两个模型的权重相同
        self.test_model.load_state_dict(self.ref_model.state_dict())
        self.ref_model.eval()
        self.test_model.eval()

    def test_fwd_numeric(self):
        # 测试前向传播的数值一致性
        ref_output = self.ref_model(self.x)
        test_output = self.test_model(self.x)
        
        # 使用assertAllClose来检查输出是否一致
        self.assertTrue(detailed_allclose_check(ref_output[0], test_output[0], atol=1e-5, rtol=1e-5))
        self.assertTrue(detailed_allclose_check(ref_output[1], test_output[1], atol=1e-5, rtol=1e-5))
        print("fp32 numerical test passed")

    def test_speedup(self):
        # 测试速度提升
        start = time.time()
        for _ in range(100):
            self.ref_model(self.x)
        ref_time = time.time() - start
        print(f"Reference model time: {ref_time / 100 : .6f} s")

        start = time.time()
        for _ in range(100):
            self.test_model(self.x)
        test_time = time.time() - start
        print(f"Test model time: {test_time / 100 : .6f} s")
        print(f"Speedup: {ref_time / test_time : .2f}x")

        # 可以添加一个断言来确保速度提升
        self.assertLess(test_time, ref_time)

if __name__ == '__main__':
    unittest.main()