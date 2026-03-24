import argparse
import glob
import os
import random
import re
import yaml

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def get_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def clean(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


def chunkify(text, max_chars=1000):
    parts = re.split(r"(?<=[.!?])\s+", text)
    result = []
    buf = []
    buf_len = 0

    for p in parts:
        if buf_len + len(p) > max_chars and buf:
            result.append(" ".join(buf))
            buf = []
            buf_len = 0
        buf.append(p)
        buf_len += len(p)

    if buf:
        result.append(" ".join(buf))
    return result


def preprocess(in_dir, out_dir, num_shards=3):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.txt")))
    logger.info(f"{len(files)} ta fayl topildi")

    chunks = []
    for fp in files:
        logger.info(f"o'qilmoqda: {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()

        for doc in raw.split("\n\n"):
            doc = clean(doc)
            if len(doc) < 50:
                continue
            chunks.extend(chunkify(doc))

        logger.info(f"jami chunk: {len(chunks)}")

    random.shuffle(chunks)

    per_shard = len(chunks) // num_shards
    for i in range(num_shards):
        start = i * per_shard
        end = start + per_shard if i < num_shards - 1 else len(chunks)
        path = os.path.join(out_dir, f"shard_{i:03d}.txt")

        with open(path, "w", encoding="utf-8") as f:
            for c in chunks[start:end]:
                f.write(c + "\n")

        logger.info(f"shard {i}: {end - start} chunk -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--output", default="data/shards")
    args = ap.parse_args()

    cfg = get_config(args.config)
    preprocess(args.input, args.output, cfg["data"]["num_shards"])
