from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


SEED = 20_260_801
SAMPLES = 10_000
WEIGHTS = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])


def minmax(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(float)
    low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.ones_like(array) if high > 0 else np.zeros_like(array)
    return (array - low) / (high - low)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RASTER-Agent case robustness figure.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "mult-agent-main"
        / "outputs"
        / "runs"
        / "sini_san_liver_fibrosis_paper_short_sini_20260721_204137",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()

    scores = pd.read_csv(args.run_dir / "mechanism_scores.csv")
    ablation = pd.read_csv(args.run_dir / "ranking_ablation.csv")
    components = np.column_stack(
        [
            scores["network_centrality"].to_numpy(float),
            minmax(scores["disease_score"]),
            scores["evidence_level"].to_numpy(float),
            scores["causal_language_score"].to_numpy(float),
            minmax(scores["literature_support"]),
            scores["data_completeness"].to_numpy(float),
        ]
    )
    uncertainty = scores["uncertainty_penalty"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    multipliers = rng.uniform(0.8, 1.2, size=(SAMPLES, len(WEIGHTS)))
    sampled_weights = WEIGHTS * multipliers
    sampled_weights /= sampled_weights.sum(axis=1, keepdims=True)
    penalties = rng.uniform(0.12, 0.18, size=SAMPLES)
    sampled_scores = components @ sampled_weights.T - uncertainty[:, None] * penalties[None, :]
    sampled_ranks = rankdata(-sampled_scores, method="min", axis=0).astype(int)

    probability_rows = []
    for index, target in enumerate(scores["target"]):
        for rank in range(1, len(scores) + 1):
            probability_rows.append(
                {
                    "target": target,
                    "rank": rank,
                    "probability": float(np.mean(sampled_ranks[index] == rank)),
                }
            )
    probabilities = pd.DataFrame(probability_rows)
    stability = []
    for index, target in enumerate(scores["target"]):
        target_ranks = sampled_ranks[index]
        stability.append(
            {
                "target": target,
                "observed_rank": int(scores.loc[index, "rank"]),
                "mechanism_score": float(scores.loc[index, "mechanism_score"]),
                "top1_probability": float(np.mean(target_ranks == 1)),
                "top3_probability": float(np.mean(target_ranks <= 3)),
                "rank_q05": float(np.quantile(target_ranks, 0.05)),
                "rank_q95": float(np.quantile(target_ranks, 0.95)),
            }
        )
    stability = pd.DataFrame(stability)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(args.output_dir / "case_rank_probabilities.csv", index=False)
    stability.to_csv(args.output_dir / "case_rank_stability.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    top = scores.head(7).copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

    ax = axes[0, 0]
    colors = ["#D55E00" if rank <= 3 else "#0072B2" for rank in top["rank"]]
    ax.barh(top["target"][::-1], top["mechanism_score"][::-1], color=colors[::-1])
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Mechanism priority score")
    ax.set_title("A  Deterministic mechanism ranking", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)

    ax = axes[0, 1]
    stages = ["Network", "+Evidence", "+Direction"]
    for _, row in ablation.head(7).iterrows():
        ranks = [row["base_rank"], row["evidence_rank"], row["causal_language_rank"]]
        ax.plot(stages, ranks, marker="o", linewidth=1.2, markersize=3.5)
        ax.text(2.04, ranks[-1], row["target"], va="center", fontsize=7)
    ax.invert_yaxis()
    ax.set_ylabel("Rank")
    ax.set_xlim(-0.05, 2.48)
    ax.set_title("B  Evidence-layer rank shifts", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    ax = axes[1, 0]
    matrix = (
        probabilities[probabilities["target"].isin(top["target"])]
        .pivot(index="target", columns="rank", values="probability")
        .reindex(top["target"])
        .reindex(columns=range(1, 8), fill_value=0)
    )
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_xticks(range(7), range(1, 8))
    ax.set_xlabel("Rank under weight perturbation")
    ax.set_title("C  Rank stability, 10,000 draws", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Probability")

    ax = axes[1, 1]
    component_labels = ["Network", "Disease", "Evidence", "Direction", "Literature", "Completeness"]
    component_matrix = components[:7]
    image = ax.imshow(component_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(7), top["target"])
    ax.set_xticks(range(6), component_labels, rotation=35, ha="right")
    ax.set_title("D  Score-component profiles", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Normalized value")

    fig.subplots_adjust(left=0.10, right=0.96, top=0.94, bottom=0.11, hspace=0.38, wspace=0.35)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_dir / "case_robustness.svg", bbox_inches="tight")
    fig.savefig(args.figure_dir / "case_robustness.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
