
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat
from functools import partial

# -- extra deps --
import math
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple

# -- project deps --
from .mlps import FastLeFF,LeFF,Mlp
from .attn import Attention
from .window_attn import WindowAttention
from .window_attn_ref import WindowAttentionRefactored
from .window_attn_dnls import WindowAttentionDnls
from .window_utils import window_partition,window_reverse
from .product_attn import ProductAttention
from .l2_attn import L2Attention

def select_attn(attn_mode,sub_attn_mode):
    if attn_mode == "window":
        return select_window_attn(sub_attn_mode)
    elif attn_mode == "product":
        return select_prod_attn(sub_attn_mode)
    elif attn_mode == "stream":
        return select_prod_attn(sub_attn_mode,"stream")
    elif attn_mode == "l2":
        return select_l2_attn(sub_attn_mode)
    else:
        raise ValueError(f"Uknown window attn type [{attn_mode}]")

def select_prod_attn(sub_attn_mode):
    return partial(ProductAttention,search_fxn=sub_attn_mode)

def select_l2_attn(sub_attn_mode):
    return L2Attention

def select_window_attn(attn_mode):
    if attn_mode == "default" or attn_mode == "original":
        return WindowAttention
    elif attn_mode == "refactored":
        return WindowAttentionRefactored
    elif attn_mode == "dnls":
        return WindowAttentionDnls
    else:
        raise ValueError(f"Uknown window attn type [{attn_mode}]")

class LeWinTransformerBlockRefactored(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,norm_layer=nn.LayerNorm,
                 token_projection='linear',token_mlp='leff',
                 modulator=False,cross_modulator=False,attn_mode="window_default",
                 ps=1,pt=1,k=-1,ws=8,wt=0,stride0=1,stride1=1,dil=1,
                 nbwd=1,rbwd=False,exact=False,bs=-1,qk_frac=1.,
                 update_dists=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size*win_size, dim) # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size*win_size, dim) # cross_modulator
            self.cross_attn = Attention(dim,num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    token_projection=token_projection,)
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None

        self.norm1 = norm_layer(dim)
        # print("self.norm1: ",self.norm1)

        self.attn_mode = attn_mode
        main_attn_mode,sub_attn_mode = attn_mode.split("_")
        attn_block = select_attn(main_attn_mode,sub_attn_mode)
        if main_attn_mode == "window":
            self.attn = attn_block(
                dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, token_projection=token_projection)
        elif main_attn_mode in ["product","l2"]:
            self.attn = attn_block(
                dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, token_projection=token_projection,
                k=k, ps=ps, pt=pt, ws=ws, wt=wt, dil=dil,
                stride0=stride0, stride1=stride1,
                nbwd=nbwd, rbwd=rbwd, exact=exact, bs=bs, qk_frac=qk_frac,
                update_dists=update_dists)
        else:
            raise ValueError(f"Uknown attention mode [{attn_mode}]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn','mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)
        elif token_mlp=='leff':
            self.mlp =  LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)

        elif token_mlp=='fastleff':
            self.mlp =  FastLeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, H, W, mask=None, flows=None, state=None):
        B,T,C,H,W = x.shape
        # print("x.shape: ",x.shape)
        # B, L, C = x.shape
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))

        # -- input mask --
        # assert mask is None:
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0))\
                                 .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # -=-=-=-=-=-=-=-=-
        #    Shift Mask
        # -=-=-=-=-=-=-=-=-

        if self.shift_size > 0:
            attn_mask = self.get_shift_mask(x,attn_mask)

        # -=-=-=-=-=-=-=-=-
        #    Modulator
        # -=-=-=-=-=-=-=-=-

        if self.cross_modulator is not None:
            x_rs = x.view(B*T,C,H*W).transpose(1,2)
            shortcut = x_rs
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x, self.cross_modulator.weight)
            x = shortcut + x_cross
            x = x.transpose(1,2).view(B,T,C,H,W)

        # -=-=-=-=-=-=-=-=-
        #    Main Layer
        # -=-=-=-=-=-=-=-=-

        # -- create shortcut --
        shortcut = x

        # -- norm layer --
        x = x.view(B*T,C,H*W).transpose(1,2)
        x = self.norm1(x)
        x = x.transpose(1,2).view(B, T, C, H, W)

        # -- cyclic shift --
        if self.shift_size > 0:
            shifts = (-self.shift_size, -self.shift_size)
            shifted_x = th.roll(x,shifts=shifts,dims=(3, 4))
        else:
            shifted_x = x

        # -- run attention --
        shifted_x = self.run_attn(shifted_x,attn_mask,flows,state)

        # -- reverse cyclic shift --
        if self.shift_size > 0:
            shifts = (self.shift_size, self.shift_size)
            x = th.roll(shifted_x, shifts=shifts, dims=(3, 4))
        else:
            x = shifted_x
        # x = x.view(B*T, H * W, C)

        # -- view for ffn --
        x = x.view(B*T,C,H*W).transpose(1,2)
        shortcut = shortcut.view(B*T,C,H*W).transpose(1,2)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #    Fully Connected Layer
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- FFN --
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1,2).view(B,T,C,H,W)
        return x

    def run_attn(self,shifted_x,attn_mask,flows,state):
        if self.attn_mode == "window_default":
            return self.run_partition_attn(shifted_x,attn_mask)
        else:
            return self.run_video_attn(shifted_x,attn_mask,flows,state)

    def get_shift_mask(self,x,attn_mask):

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    interface with input shape
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        B,T,C,H,W = x.shape

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #         original
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- calculate attention mask for SW-MSA --
        shift_mask = th.zeros((1, H, W, 1)).type_as(x)
        h_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                shift_mask[:, h, w, :] = cnt
                cnt += 1
        shift_mask_windows = window_partition(shift_mask, self.win_size) # nW, win_size, win_size, 1
        shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
        shift_attn_mask = shift_mask_windows.unsqueeze(1) - \
            shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
        shift_attn_mask = shift_attn_mask.masked_fill(
            shift_attn_mask != 0,float(-100.0)).\
            masked_fill(shift_attn_mask == 0, float(0.0))
        attn_mask = attn_mask + shift_attn_mask if attn_mask is \
            not None else shift_attn_mask

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    interface with input shape
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # no reshape for attn_mask!

        return attn_mask

    def run_video_attn(self,shifted_x,attn_mask,flows,state,wsize=8):
        B,T,C,H,W = shifted_x.shape
        wmsa_in = self.apply_modulator(shifted_x,wsize)
        attn_windows = self.attn(wmsa_in, mask=attn_mask, flows=flows, state=state)
        return attn_windows

    def run_partition_attn(self,shifted_x,attn_mask):

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    interface with input shape
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- compress batch dim & swap (C <-> HW) --
        B,T,C,H,W = shifted_x.shape
        shifted_x = rearrange(shifted_x,'b t c h w-> (b t) h w c')
        # shifted_x = shifted_x.view(B*T,C,H,W)

        # -- swap C and HW --
        shifted_x = rearrange(shifted_x,'b c h w -> b h w c')

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #          original layer
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows,self.modulator.weight)
        else:
            wmsa_in = x_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #    interface with input shape
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- swap C and HW --
        shifted_x = rearrange(shifted_x,'(b t) h w c -> b t c h w',b=B)

        return shifted_x

    def apply_modulator(self,x,wsize=8):
        # -- if modular weight --
        if not(self.modulator is None):
            # print("x.shape: ",x.shape)
            b,t,c,h,w = x.shape
            mweight = self.modulator.weight
            nh,nw = h//wsize,w//wsize
            # print("mweight.shape: ",mweight.shape)
            shape_s = '(wh ww) c -> 1 1 c (rh wh) (rw ww)'
            mweight = repeat(mweight,shape_s,wh=wsize,rh=nh,rw=nw)
            # print("mweight.shape: ",mweight.shape)
            x = x + mweight
        return x

    def flops(self,H,W):
        flops = 0
        # H, W = self.input_resolution
        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H*W, self.win_size*self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H,W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


