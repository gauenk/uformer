
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat
from functools import partial

# -- extra deps --
from timm.models.layers import trunc_normal_

# -- project deps --
from .proj import InputProj,OutputProj
from .basic_uformer import BasicUformerLayer
from .basic_uformer import create_basic_enc_layer,create_basic_dec_layer
from .basic_uformer import create_basic_conv_layer
from .scaling import Downsample,Upsample
from .parse import fields2blocks
from ..utils.model_utils import apply_freeze
# from ..utils.model_utils import expand_embed_dims

class Uformer(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 depths=[2, 2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8, 16],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True,
                 modulator=False, cross_modulator=False,
                 attn_mode="default", k=-1, ps=1, pt=1, ws=8,
                 wt=0, dil=1, stride0=1, stride1=1, nbwd=1, rbwd=False,
                 exact=False, bs=-1, freeze=False,
                 embed_dim=32, **kwargs):
        super().__init__()

        # -- init --
        self.num_enc_layers = len(depths)-1
        self.num_dec_layers = len(depths)-1
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # -- our search --
        self.attn_mode = attn_mode
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt
        self.dil = dil
        self.stride0 = stride0
        self.stride1 = stride1
        self.nbwd = nbwd
        self.exact = exact
        self.bs = bs
        self.depths = depths
        self.nblocks = len(depths)
        num_encs = self.nblocks-1

        # -- unroll for each module --
        out = fields2blocks(attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                            nbwd,rbwd,exact,bs,embed_dim,freeze,
                            nblocks=self.nblocks)
        attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1 = out[:9]
        nbwd,rbwd,exact,bs,embed_dim,freeze = out[9:]
        self.freeze = freeze
        # print(embed_dim)

        # stochastic depth
        enc_dpr = [x.item() for x in th.linspace(0, drop_path_rate,
                                                    sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[-1]
        dec_dpr = enc_dpr[::-1]
        # print(enc_dpr)

        # -- reflect depths --
        depths_ref = depths + depths[:-1][::-1]
        num_heads_ref = num_heads + num_heads[0:][::-1]
        # print(depths_ref)

        # -- input/output --
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim[0],
                                    kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim[0],
                                      out_channel=in_chans, kernel_size=3, stride=1)

        # -- init partial basic layer decl --
        basic_enc_layer = partial(create_basic_enc_layer,BasicUformerLayer,embed_dim,
                                  img_size,depths_ref,num_heads_ref,win_size,
                                  self.mlp_ratio,qkv_bias,qk_scale,drop_rate,
                                  attn_drop_rate,norm_layer,use_checkpoint,
                                  token_projection,token_mlp,shift_flag,
                                  attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                                  nbwd,rbwd,num_encs,exact,bs,enc_dpr)
        basic_conv_layer = partial(create_basic_conv_layer,BasicUformerLayer,embed_dim,
                                   img_size,depths_ref,num_heads_ref,win_size,
                                   self.mlp_ratio,qkv_bias,qk_scale,drop_rate,
                                   attn_drop_rate,norm_layer,use_checkpoint,
                                   token_projection,token_mlp,shift_flag,
                                   attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                                   nbwd,rbwd,num_encs,exact,bs,conv_dpr)
        basic_dec_layer = partial(create_basic_dec_layer,BasicUformerLayer,embed_dim,
                                  img_size,depths_ref,num_heads_ref,win_size,
                                  self.mlp_ratio,qkv_bias,qk_scale,drop_rate,
                                  attn_drop_rate,norm_layer,use_checkpoint,
                                  token_projection,token_mlp,
                                  shift_flag,modulator,cross_modulator,
                                  attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                                  nbwd,rbwd,num_encs,exact,bs,dec_dpr)
        # -- info --
        # print("depths: ",depths)
        # print("drop_path[enc]: ",enc_dpr)
        # print("drop_path[dec]: ",dec_dpr)
        # print("win_size: ",win_size)
        # print("num_heads: ",num_heads)

        # -- encoder --
        enc_list = []
        for l_enc in range(num_encs):

            # -- decl --
            mod_in = num_heads[l_enc]
            mod_out = num_heads[l_enc+1]
            setattr(self,"encoderlayer_%d" % l_enc,basic_enc_layer(l_enc))
            setattr(self,"dowsample_%d" % l_enc,dowsample(embed_dim[l_enc]*mod_in,
                                                          embed_dim[l_enc+1]*mod_out))
            # -- add to list --
            enc_layer = [getattr(self,"encoderlayer_%d" % l_enc),
                         getattr(self,"dowsample_%d" % l_enc)]
            enc_list.append(enc_layer)
        self.enc_list = enc_list

        # -- center --
        setattr(self,"conv",basic_conv_layer(num_encs))

        # -- decoder --
        dec_list = []
        for l_dec in range(num_encs):

            # -- decl --
            l_rev = (num_encs - 1) - l_dec
            mult = 1 if l_dec==0 else 2
            mod_in,mod_out = num_heads[l_rev+1],num_heads[l_rev]
            setattr(self,"upsample_%d" % l_dec,upsample(mult*embed_dim[l_rev+1]*mod_in,
                                                        embed_dim[l_rev]*mod_out))
            setattr(self,"decoderlayer_%d" % l_dec,basic_dec_layer(l_dec))

            # -- add to list --
            dec_layer = [getattr(self,"upsample_%d" % l_dec),
                         getattr(self,"decoderlayer_%d" % l_dec)]
            dec_list.append(dec_layer)
        self.dec_list = dec_list

        self.apply(self._init_weights)

    def _apply_freeze(self):
        if all([f is False for f in self.freeze]): return
        apply_freeze(self,self.freeze)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @th.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @th.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, mask=None, flows=None):

        # -- Input Projection --
        b,t,c,h,w = x.shape
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # print("y.shape: ",y.shape)
        z = y
        num_encs = self.num_enc_layers

        # -- enc --
        encs = []
        for i,(enc,down) in enumerate(self.enc_list):
            _h,_w = h//(2**i),w//(2**i)
            z = enc(z,_h,_w,mask=mask)
            encs.append(z)
            z = down(z)

        # -- middle --
        mod = 2**num_encs
        _h,_w = h//mod,w//mod
        z = self.conv(z,_h,_w,mask=mask)

        # -- dec --
        for i,(up,dec) in enumerate(self.dec_list):
            i_rev = (num_encs-1)-i
            _h,_w = h//(2**(i_rev)),w//(2**(i_rev))
            z = up(z)
            z = th.cat([z,encs[i_rev]],-1)
            z = dec(z,_h,_w,mask=mask)

        # -- Output Projection --
        y = self.output_proj(z)
        print("y.shape:" ,y.shape)

        # -- residual connection --
        out = x + y if self.dd_in == 3 else y

        return out

    def flops(self,h,w):

        # -- init flops --
        flops = 0

        # -- Input Projection --
        flops += self.input_proj.flops(h,w)
        num_encs = self.num_enc_layers

        # -- enc --
        encs = []
        for i,(enc,down) in enumerate(self.enc_list):
            _h,_w = h//(2**i),w//(2**i)
            flops += enc.flops(_h,_w)
            flops += down.flops(_h,_w)

        # -- middle --
        mod = 2**num_encs
        _h,_w = h//mod,w//mod
        flops += self.conv.flops(_h,_w)

        # -- dec --
        for i,(up,dec) in enumerate(self.dec_list):
            i_rev = num_encs-1-i
            _h,_w = h//(2**(i_rev)),w//(2**(i_rev))
            flops += up.flops(_h,_w)
            flops += dec.flops(_h,_w)

        # -- Output Projection --
        flops += self.output_proj.flops(h,w)

        return flops
