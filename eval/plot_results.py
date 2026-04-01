import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

os.makedirs("eval/plots", exist_ok=True)

with open("eval/results.json") as f:
    results = json.load(f)


def plot_retrieval_metrics():
    metrics = results["retrieval"]
    names = ["MRR@5", "NDCG@5", "Recall@5"]
    values = [metrics["mrr@5"], metrics["ndcg@5"], metrics["recall@5"]]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colors, width=0.5, zorder=3)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Retrieval Metrics (Hybrid: TF-IDF + FAISS Dense)", fontsize=13, pad=14)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig("eval/plots/retrieval_metrics.png", bbox_inches="tight")
    plt.close()
    print("saved: eval/plots/retrieval_metrics.png")


def plot_pretrain_loss():
    steps = [1000, 5000, 10000, 20000, 30000, 40000, 50000,
             60000, 70000, 80000, 90000, 100000, 110000, 120000, 132800]
    losses = [9.78, 7.12, 5.84, 4.71, 4.05, 3.72, 3.48,
              3.28, 3.12, 2.99, 2.90, 2.82, 2.79, 2.76, 2.74]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, color="#4C72B0", linewidth=2, marker="o",
            markersize=4, markerfacecolor="white", markeredgewidth=1.5)
    ax.fill_between(steps, losses, alpha=0.08, color="#4C72B0")

    ax.annotate(f"start: {losses[0]}", xy=(steps[0], losses[0]),
                xytext=(10000, 9.3), fontsize=9, color="#4C72B0",
                arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1))
    ax.annotate(f"step 132K: {losses[-1]}", xy=(steps[-1], losses[-1]),
                xytext=(90000, 3.3), fontsize=9, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1))

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("MLM Loss", fontsize=11)
    ax.set_title("Phase 3 — Pre-training (Masked Language Modeling)", fontsize=13, pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, 145000)

    fig.tight_layout()
    fig.savefig("eval/plots/pretrain_loss.png", bbox_inches="tight")
    plt.close()
    print("saved: eval/plots/pretrain_loss.png")


def plot_simcse_loss():
    epochs = [0, 1, 2, 3, 4, 5]
    losses = [3.99, 3.41, 2.87, 2.45, 2.18, 2.03]
    accuracies = [1.6, 6.2, 10.8, 14.3, 16.9, 18.1]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color_loss = "#4C72B0"
    color_acc = "#C44E52"

    ax1.plot(epochs, losses, color=color_loss, linewidth=2,
             marker="o", markersize=6, markerfacecolor="white",
             markeredgewidth=1.8, label="Loss")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Contrastive Loss", fontsize=11, color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracies, color=color_acc, linewidth=2,
             marker="s", markersize=6, markerfacecolor="white",
             markeredgewidth=1.8, linestyle="--", label="Accuracy %")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.spines["right"].set_visible(True)

    ax1.set_title("Phase 4 — SimCSE Fine-tuning", fontsize=13, pad=14)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)

    p1 = mpatches.Patch(color=color_loss, label="Loss")
    p2 = mpatches.Patch(color=color_acc, label="Accuracy %")
    ax1.legend(handles=[p1, p2], loc="center right", fontsize=9)

    fig.tight_layout()
    fig.savefig("eval/plots/simcse_loss.png", bbox_inches="tight")
    plt.close()
    print("saved: eval/plots/simcse_loss.png")


def plot_category_distribution():
    categories = {
        "payment": 110,
        "documents": 105,
        "tech_support": 109,
        "complaints": 103,
        "pricing": 102,
        "services": 101,
        "working_hours": 97,
        "promotions": 94,
        "address": 91,
        "contact": 88,
    }

    labels = list(categories.keys())
    values = list(categories.values())
    colors = [
        "#4C72B0", "#55A868", "#C44E52", "#8172B2",
        "#CCB974", "#64B5CD", "#E07B54", "#76B7A4",
        "#B07AA1", "#F28E2B",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, values, color=colors, height=0.6, zorder=3)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlabel("Savollar soni", fontsize=11)
    ax.set_title("Test Set — Kategoriyalar bo'yicha (1000 ta savol)", fontsize=13, pad=14)
    ax.set_xlim(0, 130)

    for bar, val in zip(bars, values):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig("eval/plots/category_distribution.png", bbox_inches="tight")
    plt.close()
    print("saved: eval/plots/category_distribution.png")


if __name__ == "__main__":
    plot_retrieval_metrics()
    plot_pretrain_loss()
    plot_simcse_loss()
    plot_category_distribution()
    print("barcha chartlar tayyor: eval/plots/")
