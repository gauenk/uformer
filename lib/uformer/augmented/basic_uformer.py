
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
                 modulator=False,cross_modulator=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        # lewin_type == "default" if wattn_type == "
        # lewin_block = LeWinTransformerBlock
        lewin_block = LeWinTransformerBlockRefactored
        if shift_flag:
            self.blocks = nn.ModuleList([
                lewin_block(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0 if (i % 2 == 0) else win_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    modulator=modulator,cross_modulator=cross_modulator)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                lewin_block(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    modulator=modulator,cross_modulator=cross_modulator)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops
