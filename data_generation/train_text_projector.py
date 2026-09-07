"""Stage 6 (optional): train the BERT -> motion-latent projector used by DIMO's language mode.

    python train_text_projector.py object.name=trump projector.checkpoint_dir=../outputs/trump/s2 \
        projector.output=../ckpts/trump/mlp_encoder.pth

Requires a trained DIMO model (its ``latent_codes.pth``) and the dataset's ``captions.json``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datagen.config import load_config
from datagen.projector import train_projector


def main():
    cfg = load_config(description=__doc__)
    p = cfg.projector
    if not p.checkpoint_dir:
        raise SystemExit("`projector.checkpoint_dir` must point to a DIMO checkpoint folder (e.g. outputs/<object>/s2)")
    # `dataset.name or object.name` matches where build_dataset.py writes.
    dataset_dir = os.path.join(cfg.dataset.output_dir, cfg.dataset.name or cfg.object.name)
    output = p.output or os.path.join(p.checkpoint_dir, "mlp_encoder.pth")
    train_projector(dataset_dir, p.checkpoint_dir, output, step=p.step, iters=p.iters, lr=p.lr, noise_std=p.noise_std,
                    hidden_size=p.hidden_size, num_layers=p.num_layers, bert_cache_dir=p.bert_cache_dir, seed=cfg.seed,
                    device=cfg.device)


if __name__ == "__main__":
    main()
