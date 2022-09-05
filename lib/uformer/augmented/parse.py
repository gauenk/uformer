
from ..utils.model_keys import translate_values

def fields2blocks(attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,
                  nbwd,rbwd,exact,bs,nblocks=5):
    # return attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,nbwd,exact
    order = ["attn_mode","k","ps","pt","ws","wt","dil",
             "stride0","stride1","nbwd","rbwd","exact","bs"]
    fields = {"attn_mode":attn_mode, "k":k, "ps":ps, "pt":pt,
              "ws":ws, "wt":wt, "dil":dil, "stride0":stride1,
              "stride1":stride0, "nbwd":nbwd, "rbwd": rbwd,
              "exact":exact,"bs":bs,
    }
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

