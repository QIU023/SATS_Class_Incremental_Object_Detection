from .my_vit import ViT
import torch
import torch.nn as nn

from .feature_rescale import Feature_rescale_module


# ONLY USING VIT BASE

class ViTDet_backbone(nn.Module):
    def __init__(self, vit_backbone, final_in, final_out, h, w):
        self.vit_backbone = vit_backbone
        self.feature_rescale_module = Feature_rescale_module(final_in, final_out, h, w)

    def forward(self, x):
        x, attn = self.vit_backbone(x)
        fpn = self.feature_rescale_module(x)
        return fpn, attn, x



