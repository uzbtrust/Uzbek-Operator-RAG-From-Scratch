import argparse
import json
import os
import sys
import yaml
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retriever.chunker import load_and_chunk
from retriever.hybrid_retriever import HybridRetriever
from rag.confidence import ConfidenceChecker


def mrr_at_k(relevant_positions, k=5):
    scores = []
    for pos in relevant_positions:
        if pos is not None and pos < k:
            scores.append(1.0 / (pos + 1))
        else:
            scores.append(0.0)
    return np.mean(scores) if scores else 0.0


def ndcg_at_k(relevant_positions, k=5):
    scores = []
    for pos in relevant_positions:
        if pos is not None and pos < k:
            dcg = 1.0 / np.log2(pos + 2)
            scores.append(dcg)
        else:
            scores.append(0.0)
    ideal = 1.0
    return np.mean([s / ideal for s in scores]) if scores else 0.0


def recall_at_k(relevant_positions, k=5):
    hits = sum(1 for p in relevant_positions if p is not None and p < k)
    return hits / len(relevant_positions) if relevant_positions else 0.0


def evaluate_retrieval(retriever, confidence_checker, qa_data, chunks, k=5):
    relevant_positions = []
    fallback_true_pos = 0
    fallback_false_pos = 0
    fallback_true_neg = 0
    fallback_false_neg = 0

    for item in qa_data:
        question = item["question"]
        category = item.get("category", "")
        is_no_info = category == "no_info"

        results = retriever.search(question, top_k=k)
        passed, score = confidence_checker.check(results)

        if is_no_info:
            if not passed:
                fallback_true_pos += 1
            else:
                fallback_false_neg += 1
            continue

        expected_context = item.get("context", "")
        found_pos = None

        for i, r in enumerate(results):
            chunk_text = r["chunk"]["text"]
            if expected_context and expected_context.lower() in chunk_text.lower():
                found_pos = i
                break

        relevant_positions.append(found_pos)

        if not passed:
            fallback_false_pos += 1
        else:
            fallback_true_neg += 1

    retrieval_metrics = {
        "mrr@5": round(mrr_at_k(relevant_positions, k), 4),
        "ndcg@5": round(ndcg_at_k(relevant_positions, k), 4),
        "recall@5": round(recall_at_k(relevant_positions, k), 4),
    }

    total_fallback = fallback_true_pos + fallback_false_pos
    total_should_fallback = fallback_true_pos + fallback_false_neg

    fallback_metrics = {
        "precision": round(fallback_true_pos / max(total_fallback, 1), 4),
        "recall": round(fallback_true_pos / max(total_should_fallback, 1), 4),
        "true_positives": fallback_true_pos,
        "false_positives": fallback_false_pos,
        "false_negatives": fallback_false_neg,
    }

    return retrieval_metrics, fallback_metrics


def run_evaluation(config_path, checkpoint_path, knowledge_path, qa_path, output_path):
    chunks = load_and_chunk(knowledge_path)
    logger.info(f"bilim bazasi: {len(chunks)} chunk")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    logger.info(f"test set: {len(qa_data)} savol")

    retriever = HybridRetriever(config_path, checkpoint_path)
    retriever.index(chunks)

    confidence = ConfidenceChecker(config_path)

    retrieval_metrics, fallback_metrics = evaluate_retrieval(
        retriever, confidence, qa_data, chunks
    )

    results = {
        "retrieval": retrieval_metrics,
        "fallback": fallback_metrics,
        "config": {
            "knowledge_file": knowledge_path,
            "qa_file": qa_path,
            "num_chunks": len(chunks),
            "num_questions": len(qa_data),
        }
    }

    logger.info("=== Retrieval Metrics ===")
    for k, v in retrieval_metrics.items():
        logger.info(f"  {k}: {v}")

    logger.info("=== Fallback Metrics ===")
    for k, v in fallback_metrics.items():
        logger.info(f"  {k}: {v}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"natijalar saqlandi: {output_path}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/simcse/best_model.pt")
    ap.add_argument("--knowledge", required=True)
    ap.add_argument("--qa-data", default="data/synthetic_qa.json")
    ap.add_argument("--output", default="evaluation_results.json")
    args = ap.parse_args()

    run_evaluation(args.config, args.checkpoint, args.knowledge, args.qa_data, args.output)
