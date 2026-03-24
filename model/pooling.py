import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooling(nn.Module):

    def forward(self, hidden, mask):
        m = mask.unsqueeze(-1).float()
        return torch.sum(hidden * m, dim=1) / torch.clamp(m.sum(dim=1), min=1e-9)


class CLSPooling(nn.Module):

    def forward(self, hidden, mask=None):
        return hidden[:, 0]


class PoolingHead(nn.Module):

    def __init__(self, d_model, mode="mean"):
        super().__init__()
        self.pooler = MeanPooling() if mode == "mean" else CLSPooling()
        self.fc = nn.Linear(d_model, d_model)
        self.act = nn.Tanh()

    def forward(self, hidden, mask):
        return self.act(self.fc(self.pooler(hidden, mask)))


class EmbeddingModel(nn.Module):

    def __init__(self, encoder, d_model, pool_mode="mean"):
        super().__init__()
        self.encoder = encoder
        self.pool = PoolingHead(d_model, pool_mode)

    def forward(self, input_ids, attention_mask=None):
        h = self.encoder(input_ids, attention_mask)
        return self.pool(h, attention_mask)

    @torch.no_grad()
    def encode(self, input_ids, attention_mask=None):
        self.eval()
        emb = self.forward(input_ids, attention_mask)
        return F.normalize(emb, p=2, dim=-1)
