
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
        shift_flag = shift_flag if self.attn_mode != "product_dnls" else False
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
                            drop_path=drop_path[i] if \
                            isinstance(drop_path,list) else drop_path,
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

def create_basic_enc_layer(base,embed_dim,img_size,depths,num_heads,win_size,
                           mlp_ratio,qkv_bias,qk_scale,drop_rate,attn_drop_rate,
                           norm_layer,use_checkpoint,token_projection,token_mlp,
                           shift_flag,attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                           nbwd,rbwd,nblocks,exact,bs,drop_path,l):
    mult = 2**l
    isize = img_size // 2**l
    print("enc: ",drop_path[sum(depths[:l]):sum(depths[:l+1])])
    layer = BasicUformerLayer(dim=embed_dim[l]*mult,
                              output_dim=embed_dim[l]*mult,
                              input_resolution=(isize,isize),
                              depth=depths[l],
                              num_heads=num_heads[l],
                              win_size=win_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=drop_path[sum(depths[:l]):sum(depths[:l+1])],
                              norm_layer=norm_layer,
                              use_checkpoint=use_checkpoint,
                              token_projection=token_projection,
                              token_mlp=token_mlp,shift_flag=shift_flag,
                              attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                              ws=ws[l], wt=wt[l], dil=dil[l],
                              stride0=stride0[l], stride1=stride1[l],
                              nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
    return layer

def create_basic_conv_layer(base,embed_dim,img_size,depths,num_heads,win_size,
                            mlp_ratio,qkv_bias,qk_scale,drop_rate,attn_drop_rate,
                            norm_layer,use_checkpoint,token_projection,token_mlp,
                            shift_flag,attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                            nbwd,rbwd,nblocks,exact,bs,drop_path,l):
    mult = 2**l
    isize = img_size // 2**l
    layer = BasicUformerLayer(dim=embed_dim[l]*mult,
                              output_dim=embed_dim[l]*mult,
                              input_resolution=(isize,isize),
                              depth=depths[nblocks],
                              num_heads=num_heads[l],
                              win_size=win_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=drop_path,
                              norm_layer=norm_layer,
                              use_checkpoint=use_checkpoint,
                              token_projection=token_projection,
                              token_mlp=token_mlp,shift_flag=shift_flag,
                              attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                              ws=ws[l], wt=wt[l], dil=dil[l],
                              stride0=stride0[l], stride1=stride1[l],
                              nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
    return layer

def create_basic_dec_layer(base,embed_dim,img_size,depths,num_heads,win_size,
                           mlp_ratio,qkv_bias,qk_scale,drop_rate,attn_drop_rate,
                           norm_layer,use_checkpoint,token_projection,token_mlp,
                           shift_flag,modulator,cross_modulator,attn_mode,
                           k,ps,pt,ws,wt,dil,stride0,stride1,
                           nbwd,rbwd,nblocks,exact,bs,drop_path,l):
    # -- size --
    _l = (nblocks - l)
    mult = 2**(_l)
    isize = img_size // mult
    print(depths)

    # -- drop paths --
    # l == 0 | dec_dpr[:depths[5]]
    # l == 1 | dec_dpr[sum(depths[5:6]):sum(depths[5:7])]
    dpr_s = nblocks + 1
    if l == 0:
        dpr = drop_path[:depths[5]]
    else:
        nbs = nblocks+1
        s = sum(depths[nbs:nbs+l])
        e = sum(depths[nbs:nbs+l+1])
        dpr = drop_path[s:e]
    # print(dpr)
    # print(depths,l,nblocks+1,nblocks+1+l)
    # print(mult)
    print(l,_l,2**(_l),mult)

    # -- init --
    layer = BasicUformerLayer(dim=embed_dim[l]*mult,
                              output_dim=embed_dim[l]*mult,
                              input_resolution=(isize,isize),
                              depth=depths[nblocks+1+l],
                              num_heads=num_heads[nblocks+1+l],
                              win_size=win_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=dpr,
                              norm_layer=norm_layer,
                              use_checkpoint=use_checkpoint,
                              token_projection=token_projection,
                              token_mlp=token_mlp,shift_flag=shift_flag,
                              modulator=modulator,cross_modulator=cross_modulator,
                              attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                              ws=ws[l], wt=wt[l], dil=dil[l],
                              stride0=stride0[l], stride1=stride1[l],
                              nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l], bs=bs[l])
    return layer

