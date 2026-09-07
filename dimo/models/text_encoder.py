"""Text -> motion-latent projection used by language-guided generation."""

from typing import List, Optional

import torch
from torch import nn

BERT_MODEL = "bert-base-cased"
MAX_WORD_LEN = 25


class MLPEncoder(nn.Module):
    """Projects a sentence embedding (e.g. BERT pooler output) to a motion latent code."""

    def __init__(self, input_size: int = 768, hidden_size: int = 128, output_size: int = 32, num_layers: int = 1):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        return self.projection(task_emb)


@torch.no_grad()
def encode_text(descriptions: List[str], cache_dir: Optional[str] = None) -> torch.Tensor:
    """BERT pooler embeddings ``(B, 768)`` of the given sentences (LIBERO-style task embeddings)."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(BERT_MODEL, cache_dir=cache_dir)
    tokens = tokenizer(
        text=descriptions,
        add_special_tokens=True,
        max_length=MAX_WORD_LEN,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    return model(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].detach()
