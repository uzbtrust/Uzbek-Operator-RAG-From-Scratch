import argparse
import glob
import os
import yaml
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def get_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def read_lines(directory, max_lines=None):
    files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    total = 0
    for fp in files:
        logger.info(f"o'qilmoqda: {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line
                total += 1
                if max_lines and total >= max_lines:
                    return
    logger.info(f"jami {total} qator o'qildi")


def train(cfg, data_dir):
    tc = cfg["tokenizer"]

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    tr = trainers.BpeTrainer(
        vocab_size=tc["vocab_size"],
        min_frequency=tc["min_frequency"],
        special_tokens=tc["special_tokens"],
        show_progress=True,
    )

    logger.info(f"BPE tokenizer o'qitilmoqda (vocab={tc['vocab_size']})...")
    tok.train_from_iterator(read_lines(data_dir, tc.get("training_corpus_size")), trainer=tr)

    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    pad_id = tok.token_to_id("[PAD]")
    tok.enable_padding(pad_id=pad_id, pad_token="[PAD]")
    tok.enable_truncation(max_length=cfg["model"]["max_seq_len"])

    save_to = tc["save_path"]
    os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
    tok.save(save_to + ".json")
    logger.info(f"saqlandi: {save_to}.json (vocab: {tok.get_vocab_size()})")

    for s in ["Hello world!", "Working hours: 9:00-18:00", "The quick brown fox"]:
        enc = tok.encode(s)
        logger.info(f"  '{s}' -> {enc.tokens[:12]}")

    return tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", default="data/shards")
    args = ap.parse_args()

    train(get_config(args.config), args.input)
