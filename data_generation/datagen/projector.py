"""Stage 6 (optional): train the text -> motion-latent projector for language-guided generation.

Short motion phrases are embedded with BERT and a linear projector is regressed onto the latent
codes of a trained DIMO model. Motions are matched by name between the dataset's ``info.json``
(whose ``input_videos`` order defines the latent-code order) and ``captions.json``.
"""

import os
from typing import Dict, List

import torch
from torch import nn

from datagen.io import ensure_dir, read_json
from datagen.text import bert_embeddings


class MLPEncoder(nn.Module):
    """Same architecture as ``dimo.models.text_encoder.MLPEncoder`` (a single linear layer by default)."""

    def __init__(self, input_size: int = 768, hidden_size: int = 128, output_size: int = 32, num_layers: int = 1):
        super().__init__()
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection(x)


def load_latents(checkpoint_dir: str, step=None) -> torch.Tensor:
    """Latent codes of a DIMO checkpoint folder (``latent_codes.pth`` or the VAE mean ``mu.pth``)."""
    suffix = f"_{step}" if step else ""
    for name in (f"latent_codes{suffix}.pth", f"mu{suffix}.pth"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu").detach().float()
    raise FileNotFoundError(f"no latent codes in {checkpoint_dir}")


def train_projector(dataset_dir: str, checkpoint_dir: str, output_path: str, step=None, iters: int = 5000, lr: float = 1e-3,
                    noise_std: float = 1e-3, hidden_size: int = 128, num_layers: int = 1, bert_cache_dir=None,
                    seed: int = 42, device: str = "cuda") -> Dict:
    torch.manual_seed(seed)
    info = read_json(os.path.join(dataset_dir, "info.json"))
    captions = read_json(os.path.join(dataset_dir, "captions.json"))
    latents = load_latents(checkpoint_dir, step)

    # Row i of the latent table belongs to the i-th motion DIMO trained on, which is only the
    # dataset's `input_videos` order when training did not override `input_videos`. Training records
    # the order it used, so prefer it: matching counts alone would let a reordered list regress every
    # phrase onto the wrong code, converge to a low loss and report nothing.
    order_path = os.path.join(checkpoint_dir, "motion_order.json")
    if os.path.exists(order_path):
        motions: List[str] = read_json(order_path)
        unknown = [m for m in motions if m not in captions]
        if unknown:
            raise ValueError(f"{len(unknown)} motions in {order_path} have no caption in "
                             f"{dataset_dir}/captions.json, starting with {unknown[:3]}")
    else:
        motions = info["input_videos"]
        print(f"[WARN] {checkpoint_dir} has no motion_order.json (checkpoint from an older run); "
              f"assuming the latent codes are in {dataset_dir}/info.json order")
    if latents.shape[0] != len(motions):
        raise ValueError(f"{latents.shape[0]} latent codes but {len(motions)} motions listed in "
                         f"{order_path if os.path.exists(order_path) else dataset_dir + '/info.json'}")
    phrases = [captions[m]["short"] for m in motions]

    embeddings = bert_embeddings(phrases, cache_dir=bert_cache_dir).to(device)
    latents = latents.to(device)
    model = MLPEncoder(768, hidden_size, latents.shape[1], num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    for it in range(iters):
        inputs = embeddings + noise_std * torch.randn_like(embeddings) if noise_std > 0 else embeddings
        loss = nn.functional.mse_loss(model(inputs), latents)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < best:
            best = loss.item()
            torch.save(model.state_dict(), output_path)
        if it % 500 == 0 or it == iters - 1:
            print(f"[INFO] iter {it}: loss {loss.item():.6f} (best {best:.6f})")
    print(f"[INFO] projector saved to {output_path}")
    return {"best_loss": best, "num_motions": len(motions), "output": output_path}
