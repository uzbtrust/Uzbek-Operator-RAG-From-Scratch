import argparse
import glob
import logging
import os
import random
import re
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.strip()
    return text


def split_into_chunks(text, max_len=1000):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > max_len and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


def process_raw_files(input_dir, output_dir, num_shards=3):
    os.makedirs(output_dir, exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    log.info(f"Found {len(raw_files)} raw files")

    all_chunks = []

    for filepath in raw_files:
        log.info(f"Processing {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for doc in f.read().split("\n\n"):
                doc = clean_text(doc)
                if len(doc) < 50:
                    continue
                chunks = split_into_chunks(doc)
                all_chunks.extend(chunks)

        log.info(f"Total chunks so far: {len(all_chunks)}")

    log.info(f"Shuffling {len(all_chunks)} chunks...")
    random.shuffle(all_chunks)

    shard_size = len(all_chunks) // num_shards
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else len(all_chunks)
        shard_path = os.path.join(output_dir, f"shard_{i:03d}.txt")

        with open(shard_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks[start:end]:
                f.write(chunk + "\n")

        log.info(f"Shard {i}: {end - start} chunks -> {shard_path}")

    log.info("Preprocessing complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/shards")
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_shards = cfg["data"]["num_shards"]
    process_raw_files(args.input, args.output, num_shards)


if __name__ == "__main__":
    main()
