import time
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

from retriever.chunker import load_and_chunk
from retriever.hybrid_retriever import HybridRetriever
from rag.confidence import ConfidenceChecker
from rag.generator import Generator


class RAGPipeline:

    def __init__(self, config_path="configs/config.yaml", dense_checkpoint=None):
        self.config_path = config_path
        self.retriever = HybridRetriever(config_path, dense_checkpoint)
        self.confidence = ConfidenceChecker(config_path)
        self.generator = Generator(config_path)
        self.chunks = []
        self.loaded_file = None

    def load_knowledge(self, txt_path):
        self.chunks = load_and_chunk(txt_path)
        self.retriever.index(self.chunks)
        self.loaded_file = txt_path
        logger.info(f"bilim bazasi yuklandi: {txt_path} ({len(self.chunks)} chunk)")

    def ask(self, question):
        start = time.time()

        if not self.chunks:
            return {
                "answer": "Hech qanday fayl yuklanmagan. Avval .txt faylni yuklang.",
                "chunks": [],
                "confidence": 0.0,
                "time": 0.0,
            }

        results = self.retriever.search(question)
        passed, score = self.confidence.check(results)

        if not passed:
            elapsed = time.time() - start
            return {
                "answer": self.confidence.get_fallback(),
                "chunks": results,
                "confidence": score,
                "time": elapsed,
            }

        chunk_texts = [r["chunk"] for r in results]
        answer = self.generator.generate(question, chunk_texts)

        elapsed = time.time() - start
        logger.info(f"javob berildi: {elapsed:.2f}s, confidence: {score:.4f}")

        return {
            "answer": answer,
            "chunks": results,
            "confidence": score,
            "time": elapsed,
        }
