# Uzbek-Operator-RAG-From-Scratch

Operator chatbot uchun noldan yozilgan RAG (Retrieval-Augmented Generation) tizimi. .txt fayldagi biznes ma'lumotlari asosida savollarga javob beradi.

## Arxitektura

- BERT-style Transformer Encoder (~50M parametr, noldan yozilgan)
- BPE tokenizer (16K vocab, noldan o'qitilgan)
- Gibrid retrieval: TF-IDF + dense embedding + score fusion
- Generator: Qwen2.5-7B-Instruct (8-bit kvantizatsiya)
- Confidence-based fallback (javob topilmasa aytadi)

## O'rnatish

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ishlatish

### Phase 1 — Ma'lumotlar

```bash
python data/download_corpus.py --max-docs 100000
python data/preprocess.py
python tokenizer/train_tokenizer.py
python data/synthetic_qa_generator.py
```

### Phase 2 — Model

`model/` papkasida transformer encoder:
- `attention.py` — Multi-head self-attention
- `transformer.py` — Encoder (8 layer, 512 hidden, 8 head)
- `mlm_head.py` — Masked language modeling
- `pooling.py` — Mean/CLS pooling

### Phase 3 — Pre-training

```bash
python training/pretrain.py --shard-id 0
```

### Phase 4 — Fine-tuning (SimCSE)

```bash
python training/finetune_simcse.py
```

### Phase 5 — RAG Pipeline

```bash
python ui/app.py --knowledge data.txt
```

## Qoidalar

- LangChain, LlamaIndex ISHLATILMAGAN
- Barcha asosiy komponentlar noldan yozilgan
- Kaggle T4 x2 (2x16GB VRAM) uchun optimizatsiya qilingan
