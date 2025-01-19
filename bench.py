import torch
import torch.nn as nn
from transformers import PretrainedConfig
import torch.nn.functional as F
from ops import mem_update, mem_update_fwd

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
            #o[:, t] = (q[:, t].view(B, H, 1, d) @ w).view(B, H, d) # store-first
            #if t % 128 == 0:
            #    print(k.norm().item(), w.norm().item(), o[:, t].norm().item())
            
        o = o.view(B, T, -1)
        o = self.out(o)
        
        #print(w.norm().item())
        
        return o
    
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
        o = mem_update_fwd(q, k, v, o, lr, w, B, T, H, d)
        
        # print(o[0, :, 0, :])
        o = o.view(B, T, -1)
        o = self.out(o)
        
        return o


if __name__ == '__main__':
    torch.manual_seed(251)
    import time
    config = Config(256, 4)
    ref_model = WorkingMemory(config).cuda()
    test_model = FastWorkingMemory(config).cuda()
    test_model.load_state_dict(ref_model.state_dict())
    x = torch.randn(4, 1024, 256).cuda()

    ref_o = ref_model(x)
    test_o = test_model(x)
    print(torch.allclose(ref_o, test_o, atol=1e-5, rtol=1e-5))
    start_time = time.time()
    for i in range(1):
        y = ref_model(x)
    end_time = time.time()

    print(y)
    print(f"Average duration per forward pass: {(end_time - start_time) / 100:.4f} seconds")
