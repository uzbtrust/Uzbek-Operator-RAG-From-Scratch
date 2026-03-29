import os
import sys
import numpy as np
import torch
import faiss
import yaml
from tokenizers import Tokenizer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import from_config
from model.pooling import EmbeddingModel


class DenseRetriever:

    def __init__(self, config_path="configs/config.yaml", checkpoint_path=None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        tok_path = self.cfg["tokenizer"]["save_path"] + ".json"
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.pad_id = self.tokenizer.token_to_id("[PAD]")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = from_config(self.cfg)
        self.model = EmbeddingModel(encoder, self.cfg["model"]["hidden_size"], pool_mode="mean")

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            logger.info(f"model yuklandi: {checkpoint_path}")
        else:
            logger.warning("checkpoint topilmadi, random weights ishlatilmoqda")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = None
        self.chunks = []

    def _encode_texts(self, texts, batch_size=32, max_len=128):
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = [self.tokenizer.encode(t) for t in batch_texts]

            ids_list = [e.ids[:max_len] for e in encoded]
            longest = max(len(ids) for ids in ids_list)

            input_ids = []
            masks = []
            for ids in ids_list:
                pad_n = longest - len(ids)
                input_ids.append(ids + [self.pad_id] * pad_n)
                masks.append([1] * len(ids) + [0] * pad_n)

            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
            masks = torch.tensor(masks, dtype=torch.long).to(self.device)

            with torch.no_grad():
                embs = self.model.encode(input_ids, masks)
            all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs)

    def build_index(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        embeddings = self._encode_texts(texts)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))

        logger.info(f"faiss index yaratildi: {len(chunks)} chunk, {dim} dim")

    def search(self, query, top_k=3):
        if self.index is None:
            return []

        q_emb = self._encode_texts([query])
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "source": "dense",
            })

        return results
