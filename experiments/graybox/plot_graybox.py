import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plot_dir = Path(__file__).parent.parent / Path("plots")
plot_dir.mkdir(parents=True, exist_ok=True)
result_dir = Path(__file__).parent.parent.parent / Path("whitebox/results")
sns.set_style("white")
MODELS = [
    "Meta-Llama-3.1-8B",
    "gpt2-xl",
    "gpt-neo-2.7b",
    "gpt-j-6B",
    "opt-2.7b",
]

METHOD_TO_LABEL = {
    "temptest": "TempTest",
    "fastdetect_gpt": "Fast-DetectGPT",
    "entropy": "Entropy",
    "logrank": "Log-rank",
    "likelihood": "Likelihood",
}

NUM_TOKENS = 50
SCORE_TEMPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

summaries = defaultdict(lambda: defaultdict(lambda: defaultdict((lambda: {}))))
for score_temp in SCORE_TEMPS:
    for model in MODELS:
        for gen_temp in [0.8]:
            for method in [
                "temptest",
                "fastdetect_gpt",
                "entropy",
                "rank",
                "logrank",
                "likelihood",
            ]:
                aurocs = []
                for dataset in ["xsum", "writing", "squad"]:
                    if method in ["entropy", "logrank", "likelihood", "rank"]:
                        with open(
                            result_dir
                            / f"{dataset}_{model}_gen{gen_temp}_{NUM_TOKENS}_{method}_results.json",
                            "r",
                        ) as f:
                            data = json.load(f)
                    else:
                        score_temp_string = (
                            "NULL" if method != "temptest" else score_temp
                        )
                        with open(
                            result_dir
                            / f"{dataset}_{model}_gen{gen_temp}_score{score_temp_string}_{NUM_TOKENS}_{method}_results.json",
                            "r",
                        ) as f:
                            data = json.load(f)

                    aurocs.append(data["auroc"])
                summaries[model][gen_temp][method][score_temp] = (
                    np.mean(aurocs),
                    np.std(aurocs),
                )

print(summaries)

# Individual plots
for model in MODELS:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    for method in ["temptest", "fastdetect_gpt", "likelihood", "entropy", "logrank"]:
        for temp in [0.8]:
            xs = SCORE_TEMPS
            ys = np.array([summaries[model][temp][method][t][0] for t in SCORE_TEMPS])
            stds = np.array([summaries[model][temp][method][t][1] for t in SCORE_TEMPS])
            ax.plot(
                xs,
                ys,
                label=f"{METHOD_TO_LABEL[method]}",
                alpha=0.6,
                marker="o",
                linewidth=5,
            )

    ax.legend()
    ax.set_xlabel("Scoring temperature", labelpad=10, fontsize=22)
    ax.set_ylabel("AUROC", labelpad=10, fontsize=22)
    ax.tick_params(axis="both", labelsize=16)
    ax.tick_params(axis="both", labelsize=16)
    leg = ax.legend(fontsize=18)

    fig.savefig(
        plot_dir / f"{model}_aurocs.png",
        bbox_inches="tight",
    )

# Aggregate plot
ys = {m: {} for m in ["temptest", "fastdetect_gpt", "likelihood", "entropy", "logrank"]}
for model in MODELS:
    for method in ["temptest", "fastdetect_gpt", "likelihood", "entropy", "logrank"]:
        for temp in [0.8]:
            xs = SCORE_TEMPS
            ys[method][model] = np.array(
                [summaries[model][temp][method][t][0] for t in SCORE_TEMPS]
            )
            stds = np.array([summaries[model][temp][method][t][1] for t in SCORE_TEMPS])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
for method in ["temptest", "fastdetect_gpt", "likelihood", "logrank"]:
    y = sum([ys[method][m] for m in ys[method]]) / len(ys[method])
    ax.plot(
        xs,
        y,
        label=f"{METHOD_TO_LABEL[method]}",
        alpha=0.6,
        marker="o",
        linewidth=5,
    )

ax.legend()
ax.set_xlabel("Scoring temperature", labelpad=10, fontsize=22)
ax.set_ylabel("AUROC", labelpad=10, fontsize=22)
ax.tick_params(axis="both", labelsize=16)
ax.tick_params(axis="both", labelsize=16)
ax.axvline(0.8, linestyle="--", linewidth=5, label="Generation temperature")
leg = ax.legend(fontsize=18)
ax.set_xticks(SCORE_TEMPS)

fig.savefig(
    plot_dir / "aggregate_aurocs.png",
    bbox_inches="tight",
)
