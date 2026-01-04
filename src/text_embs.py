from transformers import AutoTokenizer, AutoModel
from easydict import EasyDict
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Encode task embedding

    h = f(e), where
        e: pretrained task embedding from large model
        h: latent embedding (B, H)
    """

    def __init__(self, input_size=768, hidden_size=128, output_size=32, num_layers=1):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, task_emb):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(task_emb)  # (B, H)
        return h
    

def get_motion_embs(descriptions):
    """
    Bert embeddings for task embeddings. Borrow from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/lifelong/utils.py#L152.
    """
    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 25},
    })  # hardcode the config to get task embeddings according to original Libero code
    
    if cfg.task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir='/scratch/bbsh/linzhan/checkpoints/bert'
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir='/scratch/bbsh/linzhan/checkpoints/bert'
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    else:
        raise ValueError("Unsupported task embedding format")
    
    return task_embs