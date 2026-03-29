import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = "I don't have enough information to answer this question."


class ConfidenceChecker:

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.threshold = cfg["retriever"]["confidence_threshold"]
        logger.info(f"confidence threshold: {self.threshold}")

    def check(self, results):
        if not results:
            return False, 0.0

        best_score = max(r["score"] for r in results)
        passed = best_score >= self.threshold

        if not passed:
            logger.info(f"confidence past: {best_score:.4f} < {self.threshold}")

        return passed, best_score

    def get_fallback(self):
        return FALLBACK_RESPONSE
