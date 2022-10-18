
# -- api --
from . import original
from . import augmented
from . import warmup_scheduler
from . import configs
from . import exps
from . import flow

# -- for loading model --
from .utils.misc import optional
from .augmented import extract_model_io # set input params

def load_model(*args,**kwargs):
    attn_mode = optional(kwargs,"attn_mode","product_dnls")
    if attn_mode == "original":
        return original.load_model(*args,**kwargs)
    else:
        return augmented.load_model(*args,**kwargs)
