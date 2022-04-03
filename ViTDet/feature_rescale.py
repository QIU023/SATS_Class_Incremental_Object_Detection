import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Identity

def get32x_ds():
    return nn.AvgPool2d(kernel_size=2, stride=2)

def get8x_ups(in_chn, out_chn, h, w):
    return  nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2)

def get4x_ups(in_chn, out_chn, h, w):
    return nn.Sequential( 
        nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2),
        nn.LayerNorm([2*h, 2*w]),
        nn.GELU(),
        nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2)
    )

def get_fpn_process(in_chn, out_chn, h, w):
    return nn.Sequential(
        nn.Conv2d(in_chn, 256, kernel_size=1, stride=1),
        nn.LayerNorm([h,w]),
        nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
        nn.LayerNorm([h,w])
    )

class Feature_rescale_module(nn.Module):
    def __init__(self, in_chn, out_chn, h, w):
        self.scale_list = [get4x_ups(in_chn, out_chn, h, w),  get8x_ups(in_chn, out_chn, h, w), nn.Identity(), get32x_ds()]
        # self.up8x =
        # self.down32x = get32x_ds()
        self.fpn_process = nn.ModuleList([get_fpn_process(in_chn, out_chn, h, w)]*4)

    def forward(self, x):
        out = []
        for i in range(4):
            out.append(self.fpn_process[i](self.scale_list[i](x)))
        return out


