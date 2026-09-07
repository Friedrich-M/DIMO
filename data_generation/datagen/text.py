"""BERT sentence embeddings (shared with DIMO's language-guided generation) and a small k-means."""

from typing import List, Optional

import numpy as np
import torch

BERT_MODEL = "bert-base-cased"
MAX_WORD_LEN = 25


@torch.no_grad()
def bert_embeddings(texts: List[str], cache_dir: Optional[str] = None, device: str = "cpu") -> torch.Tensor:
    """Pooler outputs ``(N, 768)`` of ``bert-base-cased``; identical to ``dimo.models.text_encoder.encode_text``."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(BERT_MODEL, cache_dir=cache_dir).to(device).eval()
    # `truncation` is deliberately off, matching dimo.models.text_encoder.encode_text: truncating
    # here would train the projector on a different embedding than inference produces for the same
    # phrase. Phrases longer than MAX_WORD_LEN tokens simply keep their full length.
    tokens = tokenizer(text=texts, add_special_tokens=True, max_length=MAX_WORD_LEN, padding="max_length",
                       return_attention_mask=True, return_tensors="pt").to(device)
    return model(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].cpu()


def kmeans_representatives(features: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> List[int]:
    """Indices of the ``k`` samples closest to k-means centroids (deterministic given ``seed``)."""
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    k = min(k, n)
    centers = features[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        dist = ((features[:, None, :] - centers[None]) ** 2).sum(-1)
        assign = dist.argmin(1)
        new_centers = np.stack([features[assign == j].mean(0) if np.any(assign == j) else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    dist = ((features[:, None, :] - centers[None]) ** 2).sum(-1)
    chosen = []
    for j in range(k):
        order = np.argsort(dist[:, j])
        chosen.append(int(next(i for i in order if i not in chosen)))
    return sorted(chosen)
