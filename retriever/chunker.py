import re
import os
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def looks_like_kv(line):
    return bool(re.match(r'^[^:]{2,40}:\s*.+', line))


def looks_like_list_item(line):
    return bool(re.match(r'^[\-\*\•]\s+.+', line)) or bool(re.match(r'^\d+[\.\)]\s+.+', line))


def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
    stopwords = {"the", "and", "for", "that", "this", "with", "are", "you", "can", "our", "has", "have", "from"}
    return list(set(w for w in words if w not in stopwords))[:10]


def classify_chunk(lines):
    kv_count = sum(1 for l in lines if looks_like_kv(l))
    list_count = sum(1 for l in lines if looks_like_list_item(l))

    if kv_count >= len(lines) * 0.5:
        return "key_value"
    if list_count >= len(lines) * 0.5:
        return "list"
    return "paragraph"


def chunk_text(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunks = []
    current = []
    position = 0

    for line in lines:
        is_header = (
            line.endswith(":") and len(line) < 60
            or re.match(r'^[A-Z][A-Z\s]{3,}$', line)
            or re.match(r'^#{1,3}\s+', line)
        )

        if is_header and current:
            text_block = " ".join(current)
            chunk_type = classify_chunk(current)
            chunks.append({
                "text": text_block,
                "type": chunk_type,
                "position": position,
                "keywords": extract_keywords(text_block),
            })
            position += 1
            current = [line]
        else:
            current.append(line)

            if len(" ".join(current)) > 800:
                text_block = " ".join(current)
                chunk_type = classify_chunk(current)
                chunks.append({
                    "text": text_block,
                    "type": chunk_type,
                    "position": position,
                    "keywords": extract_keywords(text_block),
                })
                position += 1
                current = []

    if current:
        text_block = " ".join(current)
        chunk_type = classify_chunk(current)
        chunks.append({
            "text": text_block,
            "type": chunk_type,
            "position": position,
            "keywords": extract_keywords(text_block),
        })

    return chunks


def load_and_chunk(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_text(content)
    logger.info(f"{len(chunks)} ta chunk yaratildi: {txt_path}")
    return chunks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/chunks.json")
    args = ap.parse_args()

    chunks = load_and_chunk(args.input)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"saqlandi: {args.output}")
    for c in chunks[:3]:
        logger.info(f"  [{c['type']}] {c['text'][:80]}...")
