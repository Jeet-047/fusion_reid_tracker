import torch
import torch.nn as nn
from .attention_blocks import LocalRefinementUnit, HeterogeneousTransmissionModule, GeM

class FusionReID(nn.Module):
    def __init__(self, cnn_backbone, transformer_backbone, dim=768, num_htm_layers=2):
        super().__init__()
        self.cnn = cnn_backbone
        self.vit = transformer_backbone
        self.lru = LocalRefinementUnit(cnn_dim=2048, vit_dim=768)
        self.htm_stack = nn.ModuleList([
            HeterogeneousTransmissionModule(cnn_dim=2048, vit_dim=768) for _ in range(num_htm_layers)
        ])
        self.gem_pool = GeM()

    def forward(self, image):
        f_cnn = self.cnn(image)        # [B, C, H, W]
        f_vit = self.vit(image)        # [B, N+1, D]
        if f_vit.dim() == 2:
            f_vit = f_vit.unsqueeze(0)

        # Align features
        f_cnn, f_vit = self.lru(f_cnn, f_vit)

        # Pass through HTM
        for htm in self.htm_stack:
            f_cnn, f_vit = htm(f_cnn, f_vit)

        # Pooling
        embed_cnn = self.gem_pool(f_cnn)
        embed_vit = self.gem_pool(f_vit)
        return torch.cat([embed_cnn, embed_vit], dim=1)  # final embedding
