import os
import sys
import argparse
import tempfile
import yaml
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.pipeline import RAGPipeline

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

pipeline = None


def init_pipeline(config_path, dense_checkpoint):
    global pipeline
    pipeline = RAGPipeline(config_path, dense_checkpoint)
    logger.info("pipeline tayyor")


def upload_file(file):
    if file is None:
        return "Fayl yuklanmadi"

    pipeline.load_knowledge(file.name)
    n = len(pipeline.chunks)
    return f"Yuklandi: {os.path.basename(file.name)} ({n} ta chunk)"


def ask_question(question, history):
    if pipeline is None:
        return history + [[question, "Pipeline yuklanmadi"]]

    if not pipeline.chunks:
        return history + [[question, "Avval .txt faylni yuklang"]]

    result = pipeline.ask(question)

    answer = result["answer"]
    conf = result["confidence"]
    elapsed = result["time"]

    chunks_text = ""
    for i, r in enumerate(result["chunks"], 1):
        chunks_text += f"\n--- Chunk {i} (score: {r['score']:.4f}, source: {r['source']}) ---\n"
        chunks_text += r["chunk"]["text"][:300]
        chunks_text += "\n"

    full_answer = f"{answer}\n\n---\nConfidence: {conf:.4f} | Time: {elapsed:.2f}s"

    if chunks_text:
        full_answer += f"\n\nRetrieved chunks:{chunks_text}"

    return history + [[question, full_answer]]


def build_ui():
    with gr.Blocks(title="Operator Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Operator Assistant\nUpload a .txt file and ask questions about it.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload .txt", file_types=[".txt"])
                upload_status = gr.Textbox(label="Status", interactive=False)
                file_input.change(upload_file, inputs=file_input, outputs=upload_status)

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=500)
                msg = gr.Textbox(label="Savolingizni yozing", placeholder="What are the working hours?")
                msg.submit(ask_question, inputs=[msg, chatbot], outputs=chatbot).then(
                    lambda: "", outputs=msg
                )

        gr.Markdown("---\n*Built from scratch. No LangChain, no LlamaIndex.*")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/simcse/best_model.pt")
    ap.add_argument("--knowledge", default=None, help=".txt fayl yo'li")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    init_pipeline(args.config, args.checkpoint)

    if args.knowledge and os.path.exists(args.knowledge):
        pipeline.load_knowledge(args.knowledge)

    app = build_ui()
    app.launch(server_port=args.port, share=True)
