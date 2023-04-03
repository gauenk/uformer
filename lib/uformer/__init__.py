
# -- api --
from . import original
from . import augmented
from . import warmup_scheduler
from . import configs
from . import exps
from . import flow
from . import lightning

# -- copmaring search (swin) --
from . import search
from .search import init_search,extract_search_config

# -- for loading model --
from .utils.misc import optional
# from .augmented import extract_config
# from .augmented import extract_config as extract_model_io # set input params
# from .augmented import extract_config as extract_model_config # aka
from .original import extract_config
from .original import extract_config as extract_model_io # set input params
from .original import extract_config as extract_model_config # aka


# -- don't use this lightning. --
from dev_basics import lightning


def load_model(*args,**kwargs):
    attn_mode = optional(kwargs,"attn_mode","product_stnls")
    # attn_mode = optional(kwargs,"attn_mode","original")
    if attn_mode == "original":
        return original.load_model(*args,**kwargs)
    else:
        return augmented.load_model(*args,**kwargs)
