import argparse
import glob
import logging
import os
import yaml
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def corpus_iterator(input_dir, max_lines=None):
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    count = 0

    for filepath in files:
        log.info(f"Reading {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
                    count += 1
                    if max_lines and count >= max_lines:
                        return

    log.info(f"Total lines fed to tokenizer: {count}")


def train_bpe(cfg, input_dir):
    tok_cfg = cfg["tokenizer"]

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=tok_cfg["vocab_size"],
        min_frequency=tok_cfg["min_frequency"],
        special_tokens=tok_cfg["special_tokens"],
        show_progress=True,
    )

    max_lines = tok_cfg.get("training_corpus_size")
    log.info(f"Training BPE tokenizer (vocab_size={tok_cfg['vocab_size']})...")

    tokenizer.train_from_iterator(
        corpus_iterator(input_dir, max_lines),
        trainer=trainer,
    )

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=cfg["model"]["max_seq_len"])

    save_path = tok_cfg["save_path"]
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    tokenizer.save(save_path + ".json")

    log.info(f"Tokenizer saved to {save_path}.json")
    log.info(f"Vocab size: {tokenizer.get_vocab_size()}")

    test_sentences = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Working hours: 9:00-18:00, Monday to Friday",
    ]

    for sent in test_sentences:
        encoded = tokenizer.encode(sent)
        log.info(f"  '{sent}' -> {encoded.tokens[:15]}...")

    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default="data/shards")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_bpe(cfg, args.input)


if __name__ == "__main__":
    main()
