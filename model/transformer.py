import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):

    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 intermediate_size, max_seq_len, dropout=0.1, gradient_checkpointing=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing

        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.embed_dropout = nn.Dropout(dropout)
        self.embed_norm = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)

        x = self.final_norm(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_encoder_from_config(cfg):
    m = cfg["model"]
    gc = cfg.get("pretraining", {}).get("gradient_checkpointing", False)

    encoder = TransformerEncoder(
        vocab_size=m["vocab_size"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        intermediate_size=m["intermediate_size"],
        max_seq_len=m["max_seq_len"],
        dropout=m["dropout"],
        gradient_checkpointing=gc,
    )

    return encoder
