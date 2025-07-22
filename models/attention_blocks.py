import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalRefinementUnit(nn.Module):
    def __init__(self, cnn_dim=2048, vit_dim=768):
        super().__init__()
        self.cnn_proj = nn.Conv2d(vit_dim, cnn_dim, 1)
        self.vit_proj = nn.Linear(cnn_dim, vit_dim)
    def forward(self, f_cnn, f_vit):
        # f_cnn: [B, C, H, W], f_vit: [B, N+1, D]
        vit_tokens = f_vit[:, 1:, :]  # [B, N, D]
        vit_pooled = vit_tokens.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        # Repeat vit_pooled to match spatial dims of f_cnn
        vit_pooled = vit_pooled.repeat(1, 1, f_cnn.shape[2], f_cnn.shape[3])  # [B, D, H, W]
        f_cnn = f_cnn + self.cnn_proj(vit_pooled)
        cnn_pooled = F.adaptive_avg_pool2d(f_cnn, 1).flatten(1)
        f_vit = f_vit + self.vit_proj(cnn_pooled).unsqueeze(1)
        return f_cnn, f_vit

class HeterogeneousTransmissionModule(nn.Module):
    def __init__(self, cnn_dim=2048, vit_dim=768):
        super().__init__()
        self.cnn2vit = nn.Conv2d(vit_dim, cnn_dim, 1)
        self.vit2cnn = nn.Linear(cnn_dim, vit_dim)
    def forward(self, f_cnn, f_vit):
        cnn_global = F.adaptive_avg_pool2d(f_cnn, 1).flatten(1)  # [B, 2048]
        vit_global = f_vit.mean(dim=1)  # [B, 768]
        f_vit = f_vit + self.vit2cnn(cnn_global).unsqueeze(1)  # [B, 1, 768] + [B, N+1, 768]
        f_cnn = f_cnn + self.cnn2vit(vit_global.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, f_cnn.shape[2], f_cnn.shape[3]))
        return f_cnn, f_vit

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        # x: [B, C, H, W] or [B, N, D]
        if x.dim() == 4:
            return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0/self.p).flatten(1)
        elif x.dim() == 3:
            return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0/self.p)
        else:
            raise ValueError('Unsupported input shape for GeM')
