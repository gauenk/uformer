
def fields2blocks(attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,nbwd,exact,nblocks=4):
    return attn_mode,k,ps,pt,ws,wt,dil,stride0,stride1,nbwd,exact
    order = ["attn_mode","k","ps","pt","ws","wt","dil",
             "stride0","stride1","nbwd","exact"]
    fields = {"attn_mode":attn_mode, "k":k, "ps":ps, "pt":pt,
              "ws":ws, "wt":wt, "dil":dil, "stride0":stride1,
              "stride1":stride0, "nbwd":nbwd, "exact":exact,
    }
    outputs = []
    for field in order:
        value_l = []
        value = fields[field]

        # -- decide if we use split --
        use_split = isinstance(value,str)
        use_split = "-" in value if field == "attn_mode" else use_split

        # -- create list of values --
        if use_split:
            values = value.split("-")
            assert len(values) == nblocks,f"Must be number of blocks [{field}]."
            value_l = [int(v) for v in values]
        else:
            value_l = [value for _ in range(nblocks)]
        outputs.append(value_l)
    print(outputs)
    return outputs
