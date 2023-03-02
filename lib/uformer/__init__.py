
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
from .augmented import extract_model_io # set input params
from .augmented import extract_model_io as extract_model_config # aka
extract_config = extract_model_config # yet another name. standardization in your future.

# -- don't use this lightning. --
from dev_basics import lightning


def load_model(*args,**kwargs):
    attn_mode = optional(kwargs,"attn_mode","product_dnls")
    if attn_mode == "original":
        return original.load_model(*args,**kwargs)
    else:
        return augmented.load_model(*args,**kwargs)
