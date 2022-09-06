
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- extra deps --
from timm.models.layers import trunc_normal_

# -- project deps --
from .proj import InputProj,OutputProj
from .basic_uformer import BasicUformerLayer
from .scaling import Downsample,Upsample
from .parse import fields2blocks

class Uformer(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True,
                 modulator=False, cross_modulator=False,
                 attn_mode="default", k=-1, ps=1, pt=1, ws=8,
                 wt=0, dil=1, stride0=1, stride1=1, nbwd=1, rbwd=False,
                 exact=False, bs=-1, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
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

        # -- unroll for each module --
        out = fields2blocks(attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                            nbwd,rbwd,exact,bs)
        attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,nbwd,rbwd,exact,bs = out

        # stochastic depth
        enc_dpr = [x.item() for x in th.linspace(0, drop_path_rate,
                                                    sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim,
                                    kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim,
                                      out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        l = 0
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)

        l = 1
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        l = 2
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)

        l = 3
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        l = 4
        self.conv = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])

        # Decoder
        l = 3
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
        self.upsample_1 = upsample(embed_dim*16, embed_dim*4)

        l = 2
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
        self.upsample_2 = upsample(embed_dim*8, embed_dim*2)

        l = 1
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
        self.upsample_3 = upsample(embed_dim*4, embed_dim)

        l = 0
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,
                            token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                            ws=ws[l], wt=wt[l], dil=dil[l],
                            stride0=stride0[l], stride1=stride1[l],
                            nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])

        self.apply(self._init_weights)

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

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None, flows=None):
        # Input Projection
        t,c,h,w = x.shape
        y = self.input_proj(x)
        y = self.pos_drop(y)

        #Encoder
        conv0 = self.encoderlayer_0(y,h,w,mask=mask)
        pool0 = self.dowsample_0(conv0)
        _h,_w = h//2,w//2
        conv1 = self.encoderlayer_1(pool0,_h,_w,mask=mask)
        pool1 = self.dowsample_1(conv1)
        _h,_w = h//(2**2),w//(2**2)
        conv2 = self.encoderlayer_2(pool1,_h,_w,mask=mask)
        pool2 = self.dowsample_2(conv2)
        _h,_w = h//(2**3),w//(2**3)
        conv3 = self.encoderlayer_3(pool2,_h,_w,mask=mask)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        _h,_w = h//(2**4),w//(2**4)
        conv4 = self.conv(pool3, _h, _w, mask=mask)

        #Decoder
        _h,_w = h//(2**3),w//(2**3)
        up0 = self.upsample_0(conv4)
        deconv0 = th.cat([up0,conv3],-1)
        deconv0 = self.decoderlayer_0(deconv0,_h,_w,mask=mask)

        _h,_w = h//(2**2),w//(2**2)
        up1 = self.upsample_1(deconv0)
        deconv1 = th.cat([up1,conv2],-1)
        deconv1 = self.decoderlayer_1(deconv1,_h,_w,mask=mask)
        del deconv0
        th.cuda.empty_cache()

        _h,_w = h//(2**1),w//(2**1)
        up2 = self.upsample_2(deconv1)
        deconv2 = th.cat([up2,conv1],-1)
        deconv2 = self.decoderlayer_2(deconv2,_h,_w,mask=mask)
        del deconv1
        th.cuda.empty_cache()

        up3 = self.upsample_3(deconv2)
        deconv3 = th.cat([up3,conv0],-1)
        deconv3 = self.decoderlayer_3(deconv3,h,w,mask=mask)
        del deconv2
        th.cuda.empty_cache()

        # Output Projection
        y = self.output_proj(deconv3)
        return x + y if self.dd_in ==3 else y

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
