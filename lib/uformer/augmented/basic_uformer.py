
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- project deps --
from .lewin import LeWinTransformerBlock
from .lewin_ref import LeWinTransformerBlockRefactored


class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn', shift_flag=True,
                 modulator=False,cross_modulator=False,attn_mode="window_dnls",
                 ps=1,pt=1,k=-1,ws=8,wt=0,stride0=1,stride1=1,dil=1,
                 nbwd=1,rbwd=False,exact=False,bs=-1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.attn_mode = attn_mode
        lewin_block = LeWinTransformerBlockRefactored
        if shift_flag:
            self.blocks = nn.ModuleList([
                lewin_block(dim=dim, input_resolution=input_resolution,
                            num_heads=num_heads, win_size=win_size,
                            shift_size=0 if (i % 2 == 0) else win_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[i] if isinstance(drop_path, list) \
                            else drop_path,
                            norm_layer=norm_layer,
                            token_projection=token_projection,token_mlp=token_mlp,
                            modulator=modulator,cross_modulator=cross_modulator,
                            attn_mode=attn_mode, k=k, ps=ps, pt=pt, ws=ws,
                            wt=wt, dil=dil, stride0=stride0, stride1=stride1,
                            nbwd=nbwd, rbwd=rbwd, exact=exact, bs=bs)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                lewin_block(dim=dim, input_resolution=input_resolution,
                            num_heads=num_heads, win_size=win_size,
                            shift_size=0,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[i] if isinstance(drop_path, \
                                                                 list) else drop_path,
                            norm_layer=norm_layer,
                            token_projection=token_projection,token_mlp=token_mlp,
                            modulator=modulator,cross_modulator=cross_modulator,
                            attn_mode=attn_mode, k=k, ps=ps, pt=pt, ws=ws,
                            wt=wt, dil=dil, stride0=stride0, stride1=stride1,
                            nbwd=nbwd, rbwd=rbwd, exact=exact, bs=bs)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, h, w, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,h,w,mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops
