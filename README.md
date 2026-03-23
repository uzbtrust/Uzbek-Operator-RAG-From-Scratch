# RAG Operator Chatbot

A Retrieval-Augmented Generation system built from scratch for an operator chatbot. Answers questions based on structured business data (contacts, hours, addresses, etc.).

## Architecture

- Custom BERT-style Transformer Encoder (~50M params)
- BPE tokenizer trained from scratch (vocab: 16K)
- Hybrid retrieval: TF-IDF + dense embeddings with score fusion
- Generation: Qwen2.5-7B-Instruct (8-bit quantized)
- Confidence-based fallback for unknown questions

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### Phase 1 — Data

```bash
python data/download_corpus.py --max-docs 100000
python data/preprocess.py
python tokenizer/train_tokenizer.py
python data/synthetic_qa_generator.py
```

### Phase 2 — Model

The transformer encoder is in `model/` with these components:
- `attention.py` — Multi-head self-attention
- `transformer.py` — Full encoder with learned positional embeddings
- `mlm_head.py` — Masked language modeling head
- `pooling.py` — Mean/CLS pooling for sentence embeddings

### Phase 3 — Pre-training

```bash
python training/pretrain.py --shard-id 0
```

### Phase 4 — Fine-tuning

```bash
python training/finetune_simcse.py
```

### Phase 5 — RAG

```bash
python ui/app.py --knowledge your_data.txt
```

## No external RAG frameworks

Everything is implemented from scratch — no LangChain, no LlamaIndex.

## Hardware

Designed for Kaggle T4 x2 (2x16GB VRAM). Supports gradient checkpointing and mixed precision.
