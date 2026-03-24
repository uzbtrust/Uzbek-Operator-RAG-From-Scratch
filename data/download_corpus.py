import argparse
import os
import yaml
from datasets import load_dataset

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def get_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def wiki_stream(cfg):
    logger.info("wikipedia yuklanmoqda...")
    ds = load_dataset("wikipedia", cfg["data"]["wiki_subset"], split="train", streaming=True)
    for row in ds:
        txt = row.get("text", "")
        if len(txt) > 100:
            yield txt


def book_stream(cfg):
    logger.info("bookcorpus yuklanmoqda...")
    try:
        ds = load_dataset(cfg["data"]["bookcorpus_name"], split="train", streaming=True)
        for row in ds:
            txt = row.get("text", "")
            if len(txt) > 50:
                yield txt
    except Exception as err:
        logger.warning(f"bookcorpus topilmadi, davom etamiz: {err}")


def download(cfg, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    shard_idx = 0
    per_shard = 500000

    def new_file(idx):
        p = os.path.join(out_dir, f"shard_{idx:03d}.txt")
        logger.info(f"yangi shard: {p}")
        return open(p, "w", encoding="utf-8")

    f = new_file(shard_idx)

    for source in [wiki_stream(cfg), book_stream(cfg)]:
        if source is None:
            continue
        for text in source:
            f.write(text.strip() + "\n\n")
            n += 1

            if n % per_shard == 0:
                f.close()
                shard_idx += 1
                f = new_file(shard_idx)

            if n % 10000 == 0:
                logger.info(f"{n} ta dokument yozildi")

            if limit and n >= limit:
                break
        if limit and n >= limit:
            break

    f.close()
    logger.info(f"tayyor. jami: {n} ta dokument")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output", default="data/raw")
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    cfg = get_config(args.config)
    download(cfg, args.output, args.max_docs)
