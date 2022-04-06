from .my_vit import ViT
import torch
import torch.nn as nn

from .feature_rescale import Feature_rescale_module


# ONLY USING VIT BASE

class ViTDet_backbone(nn.Module):
    def __init__(self, vit_config, final_in, final_out, h, w):
        self.vit_backbone = ViT(**vit_config)
        self.feature_rescale_module = Feature_rescale_module(final_in, final_out, h, w)

    def forward(self, x):
        x, attn = self.vit_backbone(x)
        fpn = self.feature_rescale_module(x)
        return fpn, attn, x

vit_config = {
    'image_size' : 256,
    'patch_size' : 32,
    'num_classes' : 1000,
    'dim' : 192,
    'depth' : 12,
    'heads' : 16,
    'mlp_dim' : 2048,
    'dropout' : 0.1,
    'emb_dropout' : 0.1,
    'window_config' : [7,7,7,7,7,None]*4,
    'final_residual_conv': [True]*4     # last layer in this block initialized to 0!!!!
}

def get_vitdet_base_backbone():



