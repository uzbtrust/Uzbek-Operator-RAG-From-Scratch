import argparse
import logging
import os
import yaml
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def stream_wikipedia(cfg):
    log.info("Loading Wikipedia EN (streaming)...")
    ds = load_dataset("wikipedia", cfg["data"]["wiki_subset"], split="train", streaming=True)
    for article in ds:
        text = article.get("text", "")
        if len(text) > 100:
            yield text


def stream_bookcorpus(cfg):
    log.info("Loading BookCorpus (streaming)...")
    try:
        ds = load_dataset(cfg["data"]["bookcorpus_name"], split="train", streaming=True)
        for row in ds:
            text = row.get("text", "")
            if len(text) > 50:
                yield text
    except Exception as e:
        log.warning(f"BookCorpus not available: {e}")
        log.warning("Continuing with Wikipedia only")


def save_raw_texts(cfg, output_dir, max_docs=None):
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    current_shard = 0
    shard_file = None
    docs_per_shard = 500000

    def open_shard(idx):
        path = os.path.join(output_dir, f"shard_{idx:03d}.txt")
        log.info(f"Writing shard {idx} -> {path}")
        return open(path, "w", encoding="utf-8")

    shard_file = open_shard(current_shard)

    for text in stream_wikipedia(cfg):
        shard_file.write(text.strip() + "\n\n")
        count += 1

        if count % docs_per_shard == 0:
            shard_file.close()
            current_shard += 1
            shard_file = open_shard(current_shard)

        if count % 10000 == 0:
            log.info(f"Processed {count} documents")

        if max_docs and count >= max_docs:
            break

    for text in stream_bookcorpus(cfg):
        shard_file.write(text.strip() + "\n\n")
        count += 1

        if count % docs_per_shard == 0:
            shard_file.close()
            current_shard += 1
            shard_file = open_shard(current_shard)

        if count % 10000 == 0:
            log.info(f"Processed {count} documents")

        if max_docs and count >= max_docs * 2:
            break

    shard_file.close()
    log.info(f"Done. Total documents saved: {count}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_raw_texts(cfg, args.output, args.max_docs)


if __name__ == "__main__":
    main()
