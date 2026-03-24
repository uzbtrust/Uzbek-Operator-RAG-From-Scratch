import torch
import torch.nn as nn


class MLMHead(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.proj.bias = self.bias

    def forward(self, h):
        return self.proj(self.norm(self.act(self.dense(h))))


class MLMModel(nn.Module):

    def __init__(self, encoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.head = MLMHead(encoder.d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.encoder(input_ids, attention_mask)
        logits = self.head(h)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}
