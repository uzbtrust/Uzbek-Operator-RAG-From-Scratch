import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a helpful operator assistant. Answer based ONLY on the provided context.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""


class Generator:

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        gc = cfg["generator"]
        model_name = gc["model_name"]
        self.max_new_tokens = gc["max_new_tokens"]
        self.temperature = gc["temperature"]

        logger.info(f"generator yuklanmoqda: {model_name}")

        quant_config = None
        if gc.get("load_in_8bit", False):
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("generator tayyor")

    def generate(self, question, chunks):
        context = "\n\n".join(c["text"] for c in chunks)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return answer
