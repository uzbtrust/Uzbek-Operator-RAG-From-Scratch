# Uzbek-Operator-RAG-From-Scratch

Operator chatbot uchun noldan yozilgan RAG (Retrieval-Augmented Generation) tizimi. .txt fayldagi biznes ma'lumotlari asosida savollarga javob beradi.

## Arxitektura

- BERT-style Transformer Encoder (~50M parametr, noldan yozilgan)
- BPE tokenizer (16K vocab, noldan o'qitilgan)
- Gibrid retrieval: TF-IDF + dense embedding + score fusion
- Generator: Qwen2.5-7B-Instruct (8-bit kvantizatsiya)
- Confidence-based fallback (javob topilmasa aytadi)

## O'rnatish (lokal)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Kaggle'da ishga tushirish

Kaggle Notebook'da **birinchi cell** shu bo'lsin — boshqa hech narsa yo'q:

```python
!git clone https://github.com/uzbtrust/Uzbek-Operator-RAG-From-Scratch.git
import os
os.chdir("Uzbek-Operator-RAG-From-Scratch")
!pip install -q datasets tokenizers pyyaml
```

Keyin alohida cell'larda:

```python
# 1. Ma'lumot yuklab olish
!python data/download_corpus.py --max-docs 200000
```

```python
# 2. Tozalash va shard qilish
!python data/preprocess.py
```

```python
# 3. Tokenizer o'qitish
!python tokenizer/train_tokenizer.py
```

```python
# 4. Pre-training (shard-id: 0, 1 yoki 2)
!python training/pretrain.py --shard-id 0
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

Masked Language Modeling bilan pre-training. Kaggle T4 GPU da 3 ta parallel sessiyada ishga tushiriladi.

```bash
# har bir Kaggle sessiyasida bitta shard
python training/pretrain.py --shard-id 0
python training/pretrain.py --shard-id 1
python training/pretrain.py --shard-id 2

# 3 ta checkpointni birlashtirish
python training/merge_checkpoints.py \
    --checkpoints checkpoints/pretrain/shard0_final.pt \
                   checkpoints/pretrain/shard1_final.pt \
                   checkpoints/pretrain/shard2_final.pt \
    --output checkpoints/pretrain/merged_model.pt
```

**Training konfiguratsiya:**

| Parametr | Qiymat |
|---|---|
| Batch size | 32 |
| Learning rate | 1e-4 |
| Warmup steps | 10,000 |
| Max steps | 500,000 |
| MLM probability | 15% |
| Precision | fp16 (mixed) |
| Optimizer | AdamW |
| Scheduler | Linear warmup + cosine decay |
| Gradient clipping | 1.0 |

**Kutilgan natijalar:**

```
MLM Loss curve (taxminiy):

Step    |  Loss
--------|-------
1,000   |  8.2
5,000   |  5.4
10,000  |  4.1
50,000  |  3.2
100,000 |  2.7
200,000 |  2.3
500,000 |  1.9
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
