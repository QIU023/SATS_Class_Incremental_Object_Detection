import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.ret_attn_ids = [10, 7, 4, 1]

    def forward(self, x):
        attn_arr = []
        idx = 0
        for attn, ff in self.layers:
            attn_out, attn_score = attn(x)
            x = attn_out + x
            x = ff(x) + x
            if idx in self.ret_attn_ids:
                attn_arr.append(attn_score)
        return x, attn_arr

cnn_hyperparam_dict = {
    # mitb0: 32 chn
    '1GF':{
        'output_chn':[3, 24, 48, 96, 192]
        # 'output_chn':[3, 48, 96, 192, 384]
    },
    '4GF':{
        
    },
    '18GF':{
        
    }
    
    
}

class ShallowCNNstem(nn.Module):
    def __init__(self, num_flops, last_chn):
        super(ShallowCNNstem, self).__init__()
        self.num_channels = cnn_hyperparam_dict[num_flops]['output_chn']
        self.conv_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_channels[ii], self.num_channels[ii+1], 3, 2, 1),
                nn.BatchNorm2d(self.num_channels[ii+1]),
                nn.ReLU(inplace=True),
                # nn.ReLU(inplace=False)
            ) for ii in range(0, len(self.num_channels)-1)
        ])
        # self.final_conv = nn.Conv2d(self.num_channels[-1], last_chn, 1, 1, 0)
    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        # x = self.final_conv(x)
        x = x.flatten(2).transpose(1, 2)
        
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # self.to_patch_embedding = ShallowCNNstem('1GF', last_chn=dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth-1, heads, dim_head, mlp_dim, dropout)

        self.out_dim = dim
        # print(dim)

        self.pool = pool
        # self.to_latent = nn.Identity()
        self.final_layernorm = nn.LayerNorm(dim)

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, attn = self.transformer(x)
        # attn = None

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.final_layernorm(x)

        # x = self.to_latent(x)
        # return {"raw_features": x,
        #         "features": x,
        #         'logits': x,
        #         "distill_attn": attn}
        return x, attn

    
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 192,
    depth = 12,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)