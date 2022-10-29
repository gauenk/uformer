from pathlib import Path
from ..utils.model_utils import filter_rel_pos,get_recent_filename

def get_pretrained_path(noise_version,optional_path):
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    if optional_path != "" and not("output" in optional_path):
        croot = fdir / "output/checkpoints/"
        mpath = get_recent_filename(croot,optional_path)
        return mpath
    elif optional_path != "":
        return optional_path
    if noise_version == "noise":
        state_fn = fdir / "weights/Uformer_sidd_B.pth"
        assert os.path.isfile(str(state_fn))
    elif noise_version == "blur":
        state_fn = fdir / "weights/Uformer_gopro_B.pth"
        assert os.path.isfile(str(state_fn))
    elif noise_version in ["rgb_noise","rain"]:
        state_fn = None
    else:
        raise ValueError(f"Uknown noise_version [{noise_version}]")
    return state_fn

