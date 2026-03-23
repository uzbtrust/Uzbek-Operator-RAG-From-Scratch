import torch
import torch.nn as nn


class MeanPooling(nn.Module):

    def forward(self, hidden_states, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask


class CLSPooling(nn.Module):

    def forward(self, hidden_states, attention_mask=None):
        return hidden_states[:, 0]


class PoolingHead(nn.Module):

    def __init__(self, hidden_size, pooling_mode="mean"):
        super().__init__()

        if pooling_mode == "mean":
            self.pooler = MeanPooling()
        elif pooling_mode == "cls":
            self.pooler = CLSPooling()
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        pooled = self.pooler(hidden_states, attention_mask)
        pooled = self.dense(pooled)
        pooled = self.activation(pooled)
        return pooled


class EmbeddingModel(nn.Module):

    def __init__(self, encoder, hidden_size, pooling_mode="mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = PoolingHead(hidden_size, pooling_mode)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.encoder(input_ids, attention_mask)
        embeddings = self.pooling(hidden, attention_mask)
        return embeddings

    def encode(self, input_ids, attention_mask=None):
        self.eval()
        with torch.no_grad():
            emb = self.forward(input_ids, attention_mask)
            emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb
