
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
        print(enc_dpr)

        # -- reflect depths --
        depths_ref = depths + depths[:-1][::-1]
        num_heads_ref = num_heads + num_heads[0:][::-1]
        print(depths_ref)

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
        for l in range(num_encs):

            # -- decl --
            setattr(self,"encoderlayer_%d" % l,basic_enc_layer(l))
            setattr(self,"dowsample_%d" % l,dowsample(embed_dim[l]*(2**l),
                                                      embed_dim[l+1]*(2**(l+1))))
            # -- add to list --
            enc_layer = [getattr(self,"encoderlayer_%d" % l),
                         getattr(self,"dowsample_%d" % l)]
            enc_list.append(enc_layer)
        self.enc_list = enc_list
            
        # -- center --
        setattr(self,"conv",basic_conv_layer(num_encs))

        # -- decoder --
        dec_list = []
        for l in range(num_encs):
            # -- decl --
            _l = num_encs - l
            if l == 0: mod_in,mod_out = 2**num_encs,2**(num_encs-1)
            else: mod_in,mod_out = 2**(_l+1),2**(_l-1)
            setattr(self,"upsample_%d" % l,upsample(embed_dim[_l]*mod_in,
                                                     embed_dim[_l-1]*mod_out))
            setattr(self,"decoderlayer_%d" % l,basic_dec_layer(l))

            # -- add to list --
            dec_layer = [getattr(self,"upsample_%d" % l),
                         getattr(self,"decoderlayer_%d" % l)]
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

    # def extra_repr(self) -> str:
    #     return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None, flows=None):

        # Input Projection
        t,c,h,w = x.shape
        y = self.input_proj(x)
        y = self.pos_drop(y)
        z = y
        nblocks = self.num_enc_layers

        # -- enc --
        encs = []
        for i,(enc,down) in enumerate(self.enc_list):
            _h,_w = h//(2**i),w//(2**i)
            z = enc(z,_h,_w,mask=mask)
            encs.append(z)
            z = down(z)

        # -- middle --
        mod = 2**nblocks
        _h,_w = h//mod,w//mod
        z = self.conv(z,_h,_w,mask=mask)

        # -- dec --
        for i,(up,dec) in enumerate(self.dec_list):
            _i = nblocks-1-i
            _h,_w = h//(2**(_i)),w//(2**(_i))
            z = up(z)
            z = th.cat([z,encs[_i]],-1)
            z = dec(z,_h,_w,mask=mask)

        # #Encoder
        # conv0 = self.encoderlayer_0(y,h,w,mask=mask)
        # # print(conv0.shape)
        # pool0 = self.dowsample_0(conv0)
        # _h,_w = h//2,w//2
        # conv1 = self.encoderlayer_1(pool0,_h,_w,mask=mask)
        # # print(conv1.shape)
        # pool1 = self.dowsample_1(conv1)
        # _h,_w = h//(2**2),w//(2**2)
        # conv2 = self.encoderlayer_2(pool1,_h,_w,mask=mask)
        # # print(conv2.shape)
        # pool2 = self.dowsample_2(conv2)
        # _h,_w = h//(2**3),w//(2**3)
        # conv3 = self.encoderlayer_3(pool2,_h,_w,mask=mask)
        # # print(conv3.shape)
        # pool3 = self.dowsample_3(conv3)

        # # -=-=-=-=-=-=-=-=-=-=-=-=-
        # # -->    Bottleneck    <--
        # # -=-=-=-=-=-=-=-=-=-=-=-=-

        # _h,_w = h//(2**4),w//(2**4)
        # conv4 = self.conv(pool3, _h, _w, mask=mask)
        # # print(conv4.shape)

        # #Decoder
        # _h,_w = h//(2**3),w//(2**3)
        # up0 = self.upsample_0(conv4)
        # deconv0 = th.cat([up0,conv3],-1)
        # deconv0 = self.decoderlayer_0(deconv0,_h,_w,mask=mask)
        # # print(deconv0.shape)

        # _h,_w = h//(2**2),w//(2**2)
        # up1 = self.upsample_1(deconv0)
        # deconv1 = th.cat([up1,conv2],-1)
        # deconv1 = self.decoderlayer_1(deconv1,_h,_w,mask=mask)
        # # print(deconv1.shape)
        # # del deconv0
        # # th.cuda.empty_cache()

        # _h,_w = h//(2**1),w//(2**1)
        # up2 = self.upsample_2(deconv1)
        # deconv2 = th.cat([up2,conv1],-1)
        # deconv2 = self.decoderlayer_2(deconv2,_h,_w,mask=mask)
        # # print(deconv2.shape)
        # # del deconv1
        # # th.cuda.empty_cache()

        # up3 = self.upsample_3(deconv2)
        # deconv3 = th.cat([up3,conv0],-1)
        # deconv3 = self.decoderlayer_3(deconv3,h,w,mask=mask)
        # print(deconv3.shape)
        # del deconv2
        # th.cuda.empty_cache()

        # Output Projection
        y = self.output_proj(z)

        # -- info --
        # print("deconv3[min,max]: ",deconv3.min().item(),deconv3.max().item())
        # print("y[min,max]: ",y.min().item(),y.max().item())
        # print("x[min,max]: ",x.min().item(),x.max().item())

        return x + y if self.dd_in == 3 else y

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso,self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops()+self.dowsample_0.flops(self.reso,self.reso)
        flops += self.encoderlayer_1.flops()+self.dowsample_1.flops(self.reso//2,self.reso//2)
        flops += self.encoderlayer_2.flops()+self.dowsample_2.flops(self.reso//2**2,self.reso//2**2)
        flops += self.encoderlayer_3.flops()+self.dowsample_3.flops(self.reso//2**3,self.reso//2**3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso//2**4,self.reso//2**4)+self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso//2**3,self.reso//2**3)+self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso//2**2,self.reso//2**2)+self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso//2,self.reso//2)+self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso,self.reso)
        return flops
