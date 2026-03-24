import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class FFN(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(self.drop(self.act(self.w1(x)))))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, d_model, n_layers, n_heads,
                 d_ff, max_len, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_norm(x)
        x = self.emb_drop(x)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)

        return self.out_norm(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def from_config(cfg):
    m = cfg["model"]
    ckpt = cfg.get("pretraining", {}).get("gradient_checkpointing", False)

    return TransformerEncoder(
        vocab_size=m["vocab_size"],
        d_model=m["hidden_size"],
        n_layers=m["num_layers"],
        n_heads=m["num_heads"],
        d_ff=m["intermediate_size"],
        max_len=m["max_seq_len"],
        dropout=m["dropout"],
        use_checkpoint=ckpt,
    )
