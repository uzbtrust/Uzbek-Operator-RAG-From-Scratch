import argparse
import os
import math
import random
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tokenizers import Tokenizer

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import from_config
from model.mlm_head import MLMModel


def get_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class TextDataset(Dataset):
    def __init__(self, shard_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = []

        logger.info(f"shard yuklanmoqda: {shard_path}")
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 20:
                    self.lines.append(line)

        logger.info(f"{len(self.lines)} ta qator yuklandi")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids[:self.max_len]
        return ids


class MLMCollator:
    def __init__(self, tokenizer, mlm_prob=0.15, max_len=512):
        self.mlm_prob = mlm_prob
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.vocab_size = tokenizer.get_vocab_size()
        self.special = {
            tokenizer.token_to_id("[PAD]"),
            tokenizer.token_to_id("[UNK]"),
            tokenizer.token_to_id("[CLS]"),
            tokenizer.token_to_id("[SEP]"),
            tokenizer.token_to_id("[MASK]"),
        }

    def __call__(self, batch):
        max_len = min(max(len(ids) for ids in batch), self.max_len)

        input_ids = []
        labels = []
        masks = []

        for ids in batch:
            ids = ids[:max_len]
            pad_len = max_len - len(ids)

            token_ids = list(ids) + [self.pad_id] * pad_len
            attn_mask = [1] * len(ids) + [0] * pad_len
            label = [-100] * max_len

            for i in range(len(ids)):
                if ids[i] in self.special:
                    continue
                if random.random() < self.mlm_prob:
                    label[i] = ids[i]
                    r = random.random()
                    if r < 0.8:
                        token_ids[i] = self.mask_id
                    elif r < 0.9:
                        token_ids[i] = random.randint(5, self.vocab_size - 1)

            input_ids.append(token_ids)
            labels.append(label)
            masks.append(attn_mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def get_cosine_schedule(optimizer, warmup, total):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg, shard_id):
    tc = cfg["tokenizer"]
    pc = cfg["pretraining"]
    mc = cfg["model"]

    tok_path = tc["save_path"] + ".json"
    if not os.path.exists(tok_path):
        logger.error(f"tokenizer topilmadi: {tok_path}")
        logger.error("avval tokenizer/train_tokenizer.py ni ishga tushiring")
        return

    tokenizer = Tokenizer.from_file(tok_path)

    shard_dir = cfg["data"]["shard_dir"]
    shard_path = os.path.join(shard_dir, f"shard_{shard_id:03d}.txt")
    if not os.path.exists(shard_path):
        logger.error(f"shard topilmadi: {shard_path}")
        return

    dataset = TextDataset(shard_path, tokenizer, mc["max_seq_len"])
    collator = MLMCollator(tokenizer, pc["mlm_probability"], mc["max_seq_len"])
    loader = DataLoader(
        dataset,
        batch_size=pc["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    encoder = from_config(cfg)
    model = MLMModel(encoder, mc["vocab_size"])
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model parametrlari: {param_count / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pc["learning_rate"],
        weight_decay=pc["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = get_cosine_schedule(optimizer, pc["warmup_steps"], pc["max_steps"])
    scaler = GradScaler(enabled=pc["fp16"])

    ckpt_dir = pc["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    total_loss = 0.0
    start_time = time.time()

    model.train()
    epoch = 0

    while global_step < pc["max_steps"]:
        epoch += 1
        logger.info(f"epoch {epoch} boshlanmoqda...")

        for batch in loader:
            if global_step >= pc["max_steps"]:
                break

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=pc["fp16"]):
                out = model(input_ids, attn_mask, labels)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            total_loss += loss.item()

            if global_step % pc["log_every"] == 0:
                avg_loss = total_loss / pc["log_every"]
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed

                logger.info(
                    f"step {global_step}/{pc['max_steps']} | "
                    f"loss: {avg_loss:.4f} | lr: {lr:.2e} | "
                    f"speed: {steps_per_sec:.1f} steps/s"
                )
                total_loss = 0.0

            if global_step % pc["save_every"] == 0:
                ckpt_path = os.path.join(ckpt_dir, f"shard{shard_id}_step{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "shard_id": shard_id,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": cfg,
                }, ckpt_path)
                logger.info(f"checkpoint saqlandi: {ckpt_path}")

    final_path = os.path.join(ckpt_dir, f"shard{shard_id}_final.pt")
    torch.save({
        "step": global_step,
        "shard_id": shard_id,
        "model": model.state_dict(),
        "config": cfg,
    }, final_path)

    elapsed = time.time() - start_time
    logger.info(f"trening tugadi. {global_step} step, {elapsed / 3600:.1f} soat")
    logger.info(f"final checkpoint: {final_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--shard-id", type=int, required=True)
    args = ap.parse_args()

    train(get_config(args.config), args.shard_id)
