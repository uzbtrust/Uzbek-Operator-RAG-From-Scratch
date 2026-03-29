import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from retriever.tfidf_retriever import TFIDFRetriever
from retriever.dense_retriever import DenseRetriever


class HybridRetriever:

    def __init__(self, config_path="configs/config.yaml", dense_checkpoint=None):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        rc = cfg["retriever"]
        self.sparse_weight = rc["sparse_weight"]
        self.dense_weight = rc["dense_weight"]
        self.top_k = rc["top_k"]

        self.tfidf = TFIDFRetriever()
        self.dense = DenseRetriever(config_path, dense_checkpoint)

        self.chunks = []

    def index(self, chunks):
        self.chunks = chunks
        self.tfidf.index(chunks)
        self.dense.build_index(chunks)
        logger.info(f"hybrid index tayyor: {len(chunks)} chunk")

    def search(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k

        fetch_k = top_k * 3

        sparse_results = self.tfidf.search(query, fetch_k)
        dense_results = self.dense.search(query, fetch_k)

        score_map = {}
        chunk_map = {}

        for r in sparse_results:
            key = r["chunk"]["position"]
            chunk_map[key] = r["chunk"]
            score_map[key] = score_map.get(key, 0) + self.sparse_weight * r["score"]

        for r in dense_results:
            key = r["chunk"]["position"]
            chunk_map[key] = r["chunk"]
            score_map[key] = score_map.get(key, 0) + self.dense_weight * r["score"]

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for pos, score in ranked:
            results.append({
                "chunk": chunk_map[pos],
                "score": score,
                "source": "hybrid",
            })

        return results
