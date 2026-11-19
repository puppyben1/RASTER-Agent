from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_source_benchmark import ALL_METRICS, bh_adjust, sha256, stable_tie_break


SEED = 20_260_801
N_BOOTSTRAP = 2_000
LOGISTIC_C = 0.3


def build_candidates(project_root: Path) -> tuple[pd.DataFrame, dict]:
    edge_dir = project_root / "TCMagent" / "network_pharmacology_edges_HERB_v2" / "results"
    compound_path = edge_dir / "compound_target.tsv"
    disease_path = edge_dir / "disease_target.tsv"
    hct_path = edge_dir / "herb_compound_target.tsv"
    evidence_path = (
        project_root
        / "TCMagent"
        / "derived_relationships_HERB_v2"
        / "results"
        / "subject_evidence_index.tsv"
    )

    all_edges = pd.read_csv(
        compound_path,
        sep="\t",
        dtype=str,
        usecols=["compound_id", "compound_name", "target_id", "target_name", "source_database"],
    )
    netpharm = all_edges[all_edges["source_database"] == "NetPharm"][
        ["compound_id", "compound_name", "target_id", "target_name"]
    ].drop_duplicates()
    stitch = all_edges[all_edges["source_database"] == "STITCH_v5.0"][
        ["compound_id", "target_id"]
    ].drop_duplicates()
    stitch["label"] = 1
    candidates = netpharm.merge(stitch, on=["compound_id", "target_id"], how="left")
    candidates["label"] = candidates["label"].fillna(0).astype(int)
    eligibility = candidates.groupby("compound_id")["label"].agg(["size", "sum"])
    eligible_ids = eligibility[
        (eligibility["sum"] >= 2) & ((eligibility["size"] - eligibility["sum"]) >= 20)
    ].index
    candidates = candidates[candidates["compound_id"].isin(eligible_ids)].copy()

    netpharm_degree = (
        netpharm.groupby("target_id", as_index=False)["compound_id"]
        .nunique()
        .rename(columns={"compound_id": "netpharm_target_compounds"})
    )
    candidates = candidates.merge(netpharm_degree, on="target_id", how="left")
    candidates["netpharm_target_compounds_excluding_query"] = np.maximum(
        0, candidates["netpharm_target_compounds"].astype(float) - 1
    )

    disease_edges = pd.read_csv(
        disease_path,
        sep="\t",
        dtype=str,
        usecols=["disease_id", "target_id", "source_database"],
    )
    hpo_context = (
        disease_edges[disease_edges["source_database"] == "HPO"]
        .groupby("target_id", as_index=False)["disease_id"]
        .nunique()
        .rename(columns={"disease_id": "hpo_disease_count"})
    )
    candidates = candidates.merge(hpo_context, on="target_id", how="left")

    hct = pd.read_csv(
        hct_path,
        sep="\t",
        dtype=str,
        usecols=["herb_id", "compound_id", "target_id"],
    )
    herb_context = (
        hct.groupby("target_id", as_index=False)
        .agg(herb_count=("herb_id", "nunique"), herb_compound_count=("compound_id", "nunique"))
    )
    candidates = candidates.merge(herb_context, on="target_id", how="left")

    evidence = pd.read_csv(
        evidence_path,
        sep="\t",
        dtype=str,
        usecols=["evidence_source", "subject_id", "subject_type"],
    )
    ingredient = evidence[evidence["subject_type"] == "Ingredient"]
    evidence_by_compound = pd.crosstab(
        ingredient["subject_id"], ingredient["evidence_source"]
    ).reset_index().rename(columns={"subject_id": "compound_id"})
    evidence_columns = ["clinical_trial", "experiment", "meta_analysis", "reference"]
    for column in evidence_columns:
        if column not in evidence_by_compound:
            evidence_by_compound[column] = 0
    evidence_by_compound["query_compound_evidence"] = evidence_by_compound[evidence_columns].sum(axis=1)
    target_evidence = (
        netpharm[["compound_id", "target_id"]]
        .merge(evidence_by_compound, on="compound_id", how="left")
        .fillna(0)
        .groupby("target_id", as_index=False)[evidence_columns + ["query_compound_evidence"]]
        .sum()
        .rename(columns={"query_compound_evidence": "target_ingredient_evidence"})
    )
    candidates = candidates.merge(target_evidence, on="target_id", how="left")
    candidates = candidates.merge(
        evidence_by_compound[["compound_id", "query_compound_evidence"]],
        on="compound_id",
        how="left",
    ).fillna(0)
    candidates["target_evidence_excluding_query"] = np.maximum(
        0,
        candidates["target_ingredient_evidence"].astype(float)
        - candidates["query_compound_evidence"].astype(float),
    )

    candidates["target_popularity"] = np.log1p(
        candidates["netpharm_target_compounds_excluding_query"].astype(float)
    )
    candidates["hpo_context"] = np.log1p(candidates["hpo_disease_count"].astype(float))
    candidates["herb_context"] = np.log1p(candidates["herb_count"].astype(float))
    candidates["evidence_context"] = np.log1p(
        candidates["target_evidence_excluding_query"].astype(float)
    )
    candidates["raster_heuristic"] = (
        0.30 * candidates["target_popularity"]
        + 0.30 * candidates["hpo_context"]
        + 0.20 * candidates["herb_context"]
        + 0.20 * candidates["evidence_context"]
    )
    candidates["random"] = stable_tie_break(candidates["compound_id"], candidates["target_id"])
    provenance = {
        "compound_target": {
            "path": str(compound_path.relative_to(project_root)),
            "sha256": sha256(compound_path),
        },
        "disease_target": {
            "path": str(disease_path.relative_to(project_root)),
            "sha256": sha256(disease_path),
        },
        "herb_compound_target": {
            "path": str(hct_path.relative_to(project_root)),
            "sha256": sha256(hct_path),
        },
        "subject_evidence_index": {
            "path": str(evidence_path.relative_to(project_root)),
            "sha256": sha256(evidence_path),
        },
    }
    return candidates.reset_index(drop=True), provenance


def fit_calibrated(candidates: pd.DataFrame) -> pd.DataFrame:
    features = [
        "netpharm_target_compounds_excluding_query",
        "hpo_disease_count",
        "herb_count",
        "herb_compound_count",
        "target_evidence_excluding_query",
    ]
    x = np.log1p(candidates[features].astype(float))
    y = candidates["label"].to_numpy(int)
    groups = candidates["compound_id"].to_numpy(str)
    candidates["raster_calibrated"] = np.nan
    fold_rows = []
    for fold, (train, test) in enumerate(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y, groups),
        start=1,
    ):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                C=LOGISTIC_C,
                solver="lbfgs",
            ),
        )
        model.fit(x.iloc[train], y[train])
        candidates.loc[test, "raster_calibrated"] = model.predict_proba(x.iloc[test])[:, 1]
        candidates.loc[test, "outer_fold"] = fold
        fold_rows.append(
            {
                "fold": fold,
                "train_compounds": int(pd.Series(groups[train]).nunique()),
                "test_compounds": int(pd.Series(groups[test]).nunique()),
                "train_candidates": len(train),
                "test_candidates": len(test),
                "train_positives": int(y[train].sum()),
                "test_positives": int(y[test].sum()),
            }
        )
    return pd.DataFrame(fold_rows)


def task_metrics(group: pd.DataFrame, method: str) -> dict:
    labels = group["label"].to_numpy(int)
    scores = group[method].to_numpy(float)
    tie_break = group["random"].to_numpy(float)
    order = np.lexsort((-tie_break, -scores))
    k = min(10, len(group))
    positive_count = int(labels.sum())
    top_k_positives = int(labels[order[:k]].sum())
    precision10 = top_k_positives / k
    prevalence = positive_count / len(group)
    first_positive = int(np.flatnonzero(labels[order])[:1][0]) + 1
    return {
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
        "ndcg10": ndcg_score([labels], [scores], k=k, ignore_ties=False),
        "recall10": float(top_k_positives / positive_count),
        "mrr": 1.0 / first_positive,
        "precision10": float(precision10),
        "enrichment10": float(precision10 / prevalence),
        "hit10": float(top_k_positives > 0),
    }


def evaluate(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = [
        "random",
        "target_popularity",
        "hpo_context",
        "herb_context",
        "evidence_context",
        "raster_heuristic",
        "raster_calibrated",
    ]
    rows = []
    for compound_id, group in candidates.groupby("compound_id"):
        for method in methods:
            rows.append(
                {
                    "compound_id": compound_id,
                    "compound_name": group["compound_name"].iloc[0],
                    "candidate_count": len(group),
                    "positive_count": int(group["label"].sum()),
                    "method": method,
                    **task_metrics(group, method),
                }
            )
    by_compound = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    summary_rows = []
    for method, group in by_compound.groupby("method"):
        for metric in ALL_METRICS:
            values = group[metric].to_numpy(float)
            bootstrap = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": float(np.quantile(bootstrap, 0.025)),
                    "ci_high": float(np.quantile(bootstrap, 0.975)),
                    "compound_count": len(values),
                }
            )
    summary = pd.DataFrame(summary_rows)

    paired_rows = []
    baselines = [
        "random",
        "target_popularity",
        "hpo_context",
        "herb_context",
        "evidence_context",
        "raster_heuristic",
    ]
    for metric in ALL_METRICS:
        full = by_compound[by_compound["method"] == "raster_calibrated"].set_index("compound_id")
        current = []
        for baseline in baselines:
            other = by_compound[by_compound["method"] == baseline].set_index("compound_id")
            difference = (full.loc[other.index, metric] - other[metric]).to_numpy(float)
            bootstrap = rng.choice(
                difference, size=(N_BOOTSTRAP, len(difference)), replace=True
            ).mean(axis=1)
            try:
                statistic, p_value = wilcoxon(difference, alternative="greater")
            except ValueError:
                statistic, p_value = np.nan, 1.0
            current.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "mean_difference": float(difference.mean()),
                    "ci_low": float(np.quantile(bootstrap, 0.025)),
                    "ci_high": float(np.quantile(bootstrap, 0.975)),
                    "win_rate": float((difference > 0).mean()),
                    "tie_rate": float((difference == 0).mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
        adjusted = bh_adjust(np.array([row["p_value"] for row in current]))
        for row, adjusted_p in zip(current, adjusted):
            row["adjusted_p"] = float(adjusted_p)
            paired_rows.append(row)
    return by_compound, summary, pd.DataFrame(paired_rows)


def make_figure(summary: pd.DataFrame, output_dir: Path) -> None:
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
    methods = ["target_popularity", "hpo_context", "herb_context", "raster_heuristic", "raster_calibrated"]
    labels = ["Target popularity", "HPO context", "Herb context", "Heuristic", "Calibrated"]
    colors = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00"]
    metrics = ["auprc", "ndcg10", "recall10"]
    metric_labels = ["AUPRC", "nDCG@10", "Recall@10"]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.15
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        selected = summary[summary["method"] == method].set_index("metric").loc[metrics]
        means = selected["mean"].to_numpy()
        errors = np.vstack(
            [means - selected["ci_low"].to_numpy(), selected["ci_high"].to_numpy() - means]
        )
        ax.bar(
            x + (index - 2) * width,
            means,
            width,
            yerr=errors,
            color=color,
            label=label,
            capsize=1.5,
            linewidth=0,
        )
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("Macro-average score")
    ax.set_ylim(0, 0.60)
    ax.set_title("A  Cross-relation evidence recovery", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    ax = axes[1]
    utility_metrics = ["precision10", "hit10"]
    utility_labels = ["Precision@10", "Compounds with at least\none hit in top 10"]
    utility_methods = ["hpo_context", "raster_calibrated"]
    utility_colors = ["#0072B2", "#D55E00"]
    x = np.arange(len(utility_metrics))
    utility_width = 0.32
    for index, (method, color) in enumerate(zip(utility_methods, utility_colors)):
        selected = summary[summary["method"] == method].set_index("metric").loc[utility_metrics]
        means = 100 * selected["mean"].to_numpy()
        upper = 100 * selected["ci_high"].to_numpy()
        errors = 100 * np.vstack(
            [means / 100 - selected["ci_low"].to_numpy(), selected["ci_high"].to_numpy() - means / 100]
        )
        bars = ax.bar(
            x + (index - 0.5) * utility_width,
            means,
            utility_width,
            yerr=errors,
            color=color,
            capsize=2,
            linewidth=0,
            label="HPO context" if method == "hpo_context" else "RASTER calibrated",
        )
        for bar, value, upper_limit in zip(bars, means, upper):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                upper_limit + 1.8,
                f"{value:.1f}%",
                ha="center",
                fontsize=7,
            )
    ax.set_xticks(x, utility_labels, fontsize=7)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 110)
    ax.set_title("B  Top-10 screening utility", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.29, wspace=0.34)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "compound_target_benchmark.svg", bbox_inches="tight")
    fig.savefig(output_dir / "compound_target_benchmark.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the source-held-out compound-target benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()

    candidates, provenance = build_candidates(args.project_root.resolve())
    folds = fit_calibrated(candidates)
    by_compound, summary, paired = evaluate(candidates)
    make_figure(summary, args.figure_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(
        args.output_dir / "compound_target_candidate_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    folds.to_csv(args.output_dir / "compound_target_folds.csv", index=False)
    by_compound.to_csv(args.output_dir / "compound_target_metrics_by_compound.csv", index=False)
    summary.to_csv(args.output_dir / "compound_target_method_summary.csv", index=False)
    paired.to_csv(args.output_dir / "compound_target_paired_comparisons.csv", index=False)
    metadata = {
        "benchmark_definition": "Rank NetPharm compound-target candidates against source-held-out STITCH v5.0 labels.",
        "stitch_excluded_from_features": True,
        "query_compound_excluded_from_target_aggregate_features": True,
        "seed": SEED,
        "compound_count": int(candidates["compound_id"].nunique()),
        "candidate_count": len(candidates),
        "positive_count": int(candidates["label"].sum()),
        "positive_prevalence": float(candidates["label"].mean()),
        "pooled_auprc": {
            method: float(average_precision_score(candidates["label"], candidates[method]))
            for method in ["hpo_context", "raster_heuristic", "raster_calibrated"]
        },
        "pooled_auprc_enrichment_over_prevalence": {
            method: float(
                average_precision_score(candidates["label"], candidates[method])
                / candidates["label"].mean()
            )
            for method in ["hpo_context", "raster_heuristic", "raster_calibrated"]
        },
        "cross_validation": "Five-fold stratified group cross-validation with compound_id as the group.",
        "bootstrap_replicates": N_BOOTSTRAP,
        "provenance": provenance,
    }
    (args.output_dir / "compound_target_benchmark_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary[summary["method"].isin(["hpo_context", "raster_heuristic", "raster_calibrated"])].to_string(index=False))


if __name__ == "__main__":
    main()
