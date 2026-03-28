import argparse
import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import from_config
from model.pooling import EmbeddingModel


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_pretrained_encoder(cfg, checkpoint_path):
    encoder = from_config(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    encoder_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            encoder_state[k.replace("encoder.", "")] = v

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        logger.warning(f"yuklanganda topilmadi: {missing[:5]}...")
    if unexpected:
        logger.warning(f"kutilmagan: {unexpected[:5]}...")

    logger.info(f"encoder yuklandi: {checkpoint_path}")
    return encoder


class QAPairDataset(Dataset):
    def __init__(self, qa_path, tokenizer, max_len=128):
        with open(qa_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.pairs = []
        for item in raw:
            if item.get("category") == "no_info":
                continue
            if not item.get("context", "").strip():
                continue
            self.pairs.append((item["question"], item["context"]))

        self.tokenizer = tokenizer
        self.max_len = max_len
        logger.info(f"{len(self.pairs)} ta QA juftlik yuklandi")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, ctx = self.pairs[idx]
        return q, ctx


class SimCSECollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def _tokenize_batch(self, texts):
        encoded = [self.tokenizer.encode(t) for t in texts]
        ids_list = [e.ids[:self.max_len] for e in encoded]
        longest = max(len(ids) for ids in ids_list)

        all_ids = []
        all_masks = []
        for ids in ids_list:
            pad_n = longest - len(ids)
            all_ids.append(ids + [self.pad_id] * pad_n)
            all_masks.append([1] * len(ids) + [0] * pad_n)

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }

    def __call__(self, batch):
        queries, contexts = zip(*batch)
        q_batch = self._tokenize_batch(queries)
        c_batch = self._tokenize_batch(contexts)
        return q_batch, c_batch


def contrastive_loss(q_embs, c_embs, temperature):
    sim = torch.mm(q_embs, c_embs.t()) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def compute_accuracy(q_embs, c_embs, temperature):
    with torch.no_grad():
        sim = torch.mm(q_embs, c_embs.t()) / temperature
        preds = sim.argmax(dim=1)
        labels = torch.arange(sim.size(0), device=sim.device)
        acc = (preds == labels).float().mean().item()
    return acc


def train(cfg, checkpoint_path, qa_path):
    fc = cfg["finetuning"]
    mc = cfg["model"]

    tok_path = cfg["tokenizer"]["save_path"] + ".json"
    tokenizer = Tokenizer.from_file(tok_path)

    encoder = load_pretrained_encoder(cfg, checkpoint_path)
    model = EmbeddingModel(encoder, mc["hidden_size"], pool_mode="mean")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"device: {device}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model parametrlari: {param_count / 1e6:.1f}M")

    dataset = QAPairDataset(qa_path, tokenizer, max_len=128)
    collator = SimCSECollator(tokenizer, max_len=128)
    loader = DataLoader(
        dataset,
        batch_size=fc["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=fc["learning_rate"],
        weight_decay=0.01,
    )

    temperature = fc["temperature"]
    save_dir = fc["save_path"]
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    start = time.time()

    for epoch in range(1, fc["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for q_batch, c_batch in loader:
            q_ids = q_batch["input_ids"].to(device)
            q_mask = q_batch["attention_mask"].to(device)
            c_ids = c_batch["input_ids"].to(device)
            c_mask = c_batch["attention_mask"].to(device)

            q_embs = model(q_ids, q_mask)
            c_embs = model(c_ids, c_mask)

            q_embs = F.normalize(q_embs, p=2, dim=-1)
            c_embs = F.normalize(c_embs, p=2, dim=-1)

            loss = contrastive_loss(q_embs, c_embs, temperature)
            acc = compute_accuracy(q_embs, c_embs, temperature)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        elapsed = time.time() - start

        logger.info(
            f"epoch {epoch}/{fc['epochs']} | "
            f"loss: {avg_loss:.4f} | acc: {avg_acc:.4f} | "
            f"vaqt: {elapsed:.0f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "loss": avg_loss,
                "config": cfg,
            }, best_path)
            logger.info(f"eng yaxshi model saqlandi: {best_path} (loss: {avg_loss:.4f})")

    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        "epoch": fc["epochs"],
        "model": model.state_dict(),
        "loss": avg_loss,
        "config": cfg,
    }, final_path)

    elapsed = time.time() - start
    logger.info(f"fine-tuning tugadi. {elapsed:.0f}s, {fc['epochs']} epoch")
    logger.info(f"final: {final_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True, help="pre-trained model .pt fayl yo'li")
    ap.add_argument("--qa-data", default="data/synthetic_qa.json")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if not os.path.exists(args.qa_data):
        logger.info("synthetic QA data topilmadi, yaratilmoqda...")
        os.system(f"python data/synthetic_qa_generator.py --output {args.qa_data}")

    train(cfg, args.checkpoint, args.qa_data)
