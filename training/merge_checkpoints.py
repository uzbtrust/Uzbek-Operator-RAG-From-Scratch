import argparse
import os
import torch
import yaml

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def get_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def merge(checkpoint_paths, output_path):
    logger.info(f"{len(checkpoint_paths)} ta checkpoint birlashtirilmoqda...")

    states = []
    for p in checkpoint_paths:
        logger.info(f"yuklanmoqda: {p}")
        ckpt = torch.load(p, map_location="cpu")
        states.append(ckpt["model"])

    merged = {}
    keys = states[0].keys()

    for key in keys:
        tensors = [s[key].float() for s in states]
        merged[key] = sum(tensors) / len(tensors)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"model": merged, "num_merged": len(states)}, output_path)
    logger.info(f"saqlandi: {output_path}")

    total_params = sum(p.numel() for p in merged.values())
    logger.info(f"jami parametrlar: {total_params / 1e6:.1f}M")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--output", default="checkpoints/pretrain/merged_model.pt")
    args = ap.parse_args()

    merge(args.checkpoints, args.output)
