import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class TFIDFRetriever:

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.chunk_matrix = None
        self.chunks = []

    def index(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.chunk_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"tfidf index yaratildi: {len(chunks)} chunk, {self.chunk_matrix.shape[1]} feature")

    def search(self, query, top_k=3):
        if self.chunk_matrix is None:
            return []

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.chunk_matrix).flatten()

        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
                "source": "tfidf",
            })

        return results
