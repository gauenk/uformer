
def run_state_search(q_vid,qstart,ntotal,k_vid,state,
                     flows,recompute_dists):
    dists,inds = stream_search(q_vid,qstart,ntotal,k_vid,state,flows,recompute_dists)
    return dists,inds

def run_recompute_dists(q_vid,k_vid,inds,fstart):
    pass

def stream_search(q_vid,qstart,ntotal,k_vid,state,flows,recompute_dists=False):
    fstart = state.fstart
    dists_new,inds_new = search_new(q_vid,qstart,ntotal,k_vid,fstart)
    if recompute_dists: dists = run_recompute_dists(q_vid,k_vid,inds,fstart)
    else: dists = state.dists
    dists = th.cat([state.dists,dists_new],0)
    inds = th.cat([state.inds,inds_new],0)
    return dists,inds

def update_state(state,dists,inds):
    if state is None: return
    elif state.type == "overlap":
        state.dists = dists
        state.inds = inds
    elif state.type == "new_frame":
        state.dists = dists
        state.inds = inds

def update_state_overlap(stat,dists,inds):
    return

def update_state_new_frame(stat,dists,inds):
    return
