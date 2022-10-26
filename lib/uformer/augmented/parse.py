
from ..utils.model_keys import translate_values
from collections import OrderedDict

def expand_embed_dim(embed_dim):
    raise NotImplementedError("")
    # -- default --
    if isinstance(embed_dim,int):
        return [[embed_dim,embed_dim] for _ in range(8)]

    # -- init --
    embed_dims = []
    _embed_dims = embed_dim.split("_")
    assert len(_embed_dims) == 6

    # -- pairs of in/out --
    for l in range(L):
        dim_l = _embed_dims[l].split("+")
        print(dim_l)
        dim_l0,dim_l1 = int(dim_l[0]),int(dim_l[1])
        embed_dims.append([dim_l0,dim_l1])
    return embed_dims

def fields2blocks(attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                  nbwd,rbwd,exact,bs,qk_frac,embed_dim,freeze,
                  update_dists,nblocks=5):
    # return attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,nbwd,exact
    # order = ["attn_mode","k","ps","pt","ws","wt","dil",
    #          "stride0","stride1","nbwd","rbwd","exact","bs","freeze"]
    fields = OrderedDict({"attn_mode":attn_mode, "k":k, "ps":ps, "pt":pt,
                          "ws":ws, "wt":wt, "dil":dil, "stride0":stride0,
                          "stride1":stride1, "nbwd":nbwd, "rbwd": rbwd,
                          "exact":exact,"bs":bs,"qk_frac":qk_frac,
                          "embed_dim":embed_dim,"freeze":freeze,
                          "update_dists":update_dists
    })
    order = list(fields.keys())
    outputs = []
    for field in order:

        # -- decide if we use split --
        value = fields[field]
        use_split = isinstance(value,str)
        use_split = ("-" in value) if field == "attn_mode" else use_split

        # -- create list of values --
        if use_split:
            values = value.split("-")
            values_l = translate_values(field,values)
            assert len(values_l) == nblocks,f"Must be number of blocks [{field}]."
        else:
            values_l = [value for _ in range(nblocks)]
        outputs.append(values_l)
    return outputs

