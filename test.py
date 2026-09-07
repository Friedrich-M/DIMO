"""Render a trained DIMO model and run the latent-space applications.

    python test.py mode=render input_folder=data/trump_n51_step20 save_path=ckpts/trump_n51_step20 \
        video_save_dir=vis/trump_n51_step20 [render_videos=11-walk] [key=value ...]

See ``configs/default.yaml`` (section "Inference") for the available modes and options.
"""

import warnings

from dimo.config import load_config
from dimo.tester import Tester

warnings.filterwarnings("ignore")


def main():
    opt = load_config(description=__doc__)
    if not opt.input_folder or not opt.save_path:
        raise SystemExit("`input_folder` and `save_path` must be set")
    Tester(opt).run()


if __name__ == "__main__":
    main()
