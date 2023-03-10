
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- project deps --
from .lewin import LeWinTransformerBlock
# from .lewin_ref import LeWinTransformerBlockRefactored

class BasicUformerLayer(nn.Module):
    def __init__(self, blocklist, block):
        super().__init__()
        self.dim = blocklist.dim
        self.input_resolution = blocklist.input_resolution
        self.depth = blocklist.depth
        self.use_checkpoint = blocklist.use_checkpoint
        self.attn_mode = blocklist.attn_mode
        Block = LeWinTransformerBlock
        self.blocks = nn.ModuleList([
                Block(block) for block in blocks])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, h, w, mask=None, flows=None, state=None):
        for blk in self.blocks:
            x = blk(x,h,w,mask,flows,state)
        return x

    def flops(self,h,w):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops(h,w)
        return flops
