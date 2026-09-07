"""Train DIMO on a folder of multi-view motion videos.

    python train.py input_folder=data/trump_n51_step20 save_path=outputs/trump_n51_step20 [key=value ...]
"""

import warnings

from dimo.config import load_config
from dimo.trainer import Trainer

warnings.filterwarnings("ignore")


def main():
    opt = load_config(description=__doc__)
    if not opt.input_folder or not opt.save_path:
        raise SystemExit("`input_folder` and `save_path` must be set")
    Trainer(opt).run()


if __name__ == "__main__":
    main()
