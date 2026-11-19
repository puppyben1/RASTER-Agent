from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20_260_801
N_BOOTSTRAP = 2_000
MIN_POSITIVES = 2
MIN_NEGATIVES = 20
LOGISTIC_C = 0.3
CORE_METRICS = ["auroc", "auprc", "ndcg10", "recall10", "mrr"]
UTILITY_METRICS = ["precision10", "enrichment10", "hit10"]
ALL_METRICS = CORE_METRICS + UTILITY_METRICS

FEATURE_GROUPS = {
    "hpo_local": ["hpo_rows", "hpo_evidence_ids", "hpo_external_ids"],
    "target_context": ["hpo_target_diseases", "hpo_target_rows"],
    "pharmacology": [
        "compound_count",
        "compound_source_count",
        "max_stitch_score",
        "herb_count",
        "hct_compound_count",
    ],
    "ingredient_evidence": [
        "clinical_trial",
        "experiment",
        "meta_analysis",
        "reference",
        "ingredient_evidence_total",
        "evidence_level",
        "literature_support",
    ],
}

METHOD_LABELS = {
    "random": "Random",
    "hpo_support": "HPO support",
    "target_popularity": "Target popularity",
    "pharm_connectivity": "Pharmacology connectivity",
    "raster_heuristic": "RASTER heuristic",
    "raster_calibrated": "RASTER calibrated",
    "raster_no_hpo_local": "RASTER without local HPO",
    "raster_no_pharmacology": "RASTER without pharmacology",
    "raster_no_ingredient_evidence": "RASTER without ingredient evidence",
}

COLORS = {
    "random": "#8C8C8C",
    "hpo_support": "#0072B2",
    "target_popularity": "#E69F00",
    "pharm_connectivity": "#009E73",
    "raster_heuristic": "#CC79A7",
    "raster_calibrated": "#D55E00",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_minmax(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("disease_id")[column]
    low = grouped.transform("min")
    high = grouped.transform("max")
    denominator = high - low
    result = pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    varying = denominator > 0
    result.loc[varying] = (frame.loc[varying, column] - low.loc[varying]) / denominator.loc[varying]
    constant_positive = (~varying) & (high > 0)
    result.loc[constant_positive] = 1.0
    return result


def stable_tie_break(disease_ids: pd.Series, target_ids: pd.Series) -> np.ndarray:
    values = []
    for disease_id, target_id in zip(disease_ids.astype(str), target_ids.astype(str)):
        token = hashlib.sha256(f"{disease_id}|{target_id}".encode("utf-8")).digest()[:8]
        values.append(int.from_bytes(token, "big") / float(2**64 - 1))
    return np.asarray(values)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0.0, 1.0)


def build_candidates(project_root: Path) -> tuple[pd.DataFrame, dict]:
    edge_dir = project_root / "TCMagent" / "network_pharmacology_edges_HERB_v2" / "results"
    evidence_path = (
        project_root
        / "TCMagent"
        / "derived_relationships_HERB_v2"
        / "results"
        / "subject_evidence_index.tsv"
    )
    disease_path = edge_dir / "disease_target.tsv"
    compound_path = edge_dir / "compound_target.tsv"
    hct_path = edge_dir / "herb_compound_target.tsv"

    disease_edges = pd.read_csv(
        disease_path,
        sep="\t",
        dtype=str,
        usecols=[
            "disease_id",
            "disease_name",
            "target_id",
            "target_name",
            "source_database",
            "evidence_id",
            "external_disease_id",
        ],
    )
    hpo = disease_edges[disease_edges["source_database"] == "HPO"].copy()
    ctd = disease_edges[disease_edges["source_database"] == "CTD"][
        ["disease_id", "target_id"]
    ].drop_duplicates()
    ctd["label"] = 1

    candidates = (
        hpo.groupby(["disease_id", "disease_name", "target_id", "target_name"], as_index=False)
        .agg(
            hpo_rows=("evidence_id", "size"),
            hpo_evidence_ids=("evidence_id", "nunique"),
            hpo_external_ids=("external_disease_id", "nunique"),
        )
        .merge(ctd, on=["disease_id", "target_id"], how="left")
    )
    candidates["label"] = candidates["label"].fillna(0).astype(int)
    eligibility = candidates.groupby("disease_id")["label"].agg(["size", "sum"])
    eligible_ids = eligibility[
        (eligibility["sum"] >= MIN_POSITIVES)
        & ((eligibility["size"] - eligibility["sum"]) >= MIN_NEGATIVES)
    ].index
    candidates = candidates[candidates["disease_id"].isin(eligible_ids)].copy()

    hpo_target = (
        hpo.groupby("target_id", as_index=False)
        .agg(
            hpo_target_diseases=("disease_id", "nunique"),
            hpo_target_rows=("evidence_id", "size"),
        )
    )
    compound_edges = pd.read_csv(
        compound_path,
        sep="\t",
        dtype=str,
        usecols=["compound_id", "target_id", "source_database", "combined_score"],
    )
    compound_edges["combined_score"] = pd.to_numeric(
        compound_edges["combined_score"], errors="coerce"
    )
    compound_features = (
        compound_edges.groupby("target_id", as_index=False)
        .agg(
            compound_count=("compound_id", "nunique"),
            compound_source_count=("source_database", "nunique"),
            max_stitch_score=("combined_score", "max"),
        )
    )
    hct = pd.read_csv(
        hct_path,
        sep="\t",
        dtype=str,
        usecols=["herb_id", "compound_id", "target_id"],
    )
    herb_features = (
        hct.groupby("target_id", as_index=False)
        .agg(
            herb_count=("herb_id", "nunique"),
            hct_compound_count=("compound_id", "nunique"),
        )
    )

    evidence = pd.read_csv(
        evidence_path,
        sep="\t",
        dtype=str,
        usecols=["evidence_source", "subject_id", "subject_type"],
    )
    ingredient_evidence = evidence[evidence["subject_type"] == "Ingredient"]
    evidence_by_compound = pd.crosstab(
        ingredient_evidence["subject_id"], ingredient_evidence["evidence_source"]
    ).reset_index().rename(columns={"subject_id": "compound_id"})
    evidence_columns = ["clinical_trial", "experiment", "meta_analysis", "reference"]
    for column in evidence_columns:
        if column not in evidence_by_compound:
            evidence_by_compound[column] = 0
    target_evidence = (
        compound_edges[["compound_id", "target_id"]]
        .drop_duplicates()
        .merge(evidence_by_compound, on="compound_id", how="inner")
        .groupby("target_id", as_index=False)[evidence_columns]
        .sum()
    )
    target_evidence["ingredient_evidence_total"] = target_evidence[evidence_columns].sum(axis=1)
    target_evidence["evidence_level"] = np.select(
        [
            target_evidence["clinical_trial"] > 0,
            target_evidence["experiment"] > 0,
            target_evidence["meta_analysis"] > 0,
            target_evidence["reference"] > 0,
        ],
        [1.0, 0.70, 0.60, 0.50],
        default=0.0,
    )
    target_evidence["literature_support"] = np.log1p(
        target_evidence["meta_analysis"] + target_evidence["reference"]
    )

    candidates = (
        candidates.merge(hpo_target, on="target_id", how="left")
        .merge(compound_features, on="target_id", how="left")
        .merge(herb_features, on="target_id", how="left")
        .merge(target_evidence, on="target_id", how="left")
        .fillna(0)
    )
    log_columns = [
        "hpo_rows",
        "hpo_evidence_ids",
        "hpo_external_ids",
        "hpo_target_diseases",
        "hpo_target_rows",
        "compound_count",
        "herb_count",
        "literature_support",
    ]
    for column in log_columns:
        candidates[f"log_{column}"] = np.log1p(candidates[column].astype(float))

    network = (
        0.45 * group_minmax(candidates, "log_hpo_target_diseases")
        + 0.25 * group_minmax(candidates, "log_compound_count")
        + 0.30 * group_minmax(candidates, "log_herb_count")
    )
    disease_relevance = (
        0.70 * group_minmax(candidates, "log_hpo_rows")
        + 0.30 * group_minmax(candidates, "log_hpo_evidence_ids")
    )
    literature = group_minmax(candidates, "literature_support")
    completeness = (
        0.40 * (candidates["compound_count"].astype(float) > 0).astype(float)
        + 0.30 * (candidates["herb_count"].astype(float) > 0).astype(float)
        + 0.30 * (candidates["ingredient_evidence_total"].astype(float) > 0).astype(float)
    )
    uncertainty = 1.0 - completeness
    candidates["hpo_support"] = candidates["log_hpo_rows"]
    candidates["target_popularity"] = candidates["log_hpo_target_diseases"]
    candidates["pharm_connectivity"] = (
        candidates["log_compound_count"] + candidates["log_herb_count"]
    )
    candidates["raster_heuristic"] = np.maximum(
        0.0,
        0.25 * network
        + 0.20 * disease_relevance
        + 0.20 * candidates["evidence_level"].astype(float)
        + 0.10 * literature
        + 0.10 * completeness
        - 0.15 * uncertainty,
    )
    candidates["random"] = stable_tie_break(candidates["disease_id"], candidates["target_id"])

    provenance = {
        "disease_target": {
            "path": str(disease_path.relative_to(project_root)),
            "sha256": sha256(disease_path),
        },
        "compound_target": {
            "path": str(compound_path.relative_to(project_root)),
            "sha256": sha256(compound_path),
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


def fit_cross_validated_models(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_features = sum(FEATURE_GROUPS.values(), [])
    feature_sets = {
        "raster_calibrated": all_features,
        "raster_no_hpo_local": [f for f in all_features if f not in FEATURE_GROUPS["hpo_local"]],
        "raster_no_pharmacology": [
            f for f in all_features if f not in FEATURE_GROUPS["pharmacology"]
        ],
        "raster_no_ingredient_evidence": [
            f for f in all_features if f not in FEATURE_GROUPS["ingredient_evidence"]
        ],
    }
    y = candidates["label"].to_numpy(int)
    groups = candidates["disease_id"].to_numpy(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows = []
    coefficient_rows = []
    for fold, (train, test) in enumerate(splitter.split(candidates, y, groups), start=1):
        candidates.loc[test, "outer_fold"] = fold
        for method, features in feature_sets.items():
            x = np.log1p(candidates[features].astype(float))
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
            candidates.loc[test, method] = model.predict_proba(x.iloc[test])[:, 1]
            fold_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "train_diseases": int(pd.Series(groups[train]).nunique()),
                    "test_diseases": int(pd.Series(groups[test]).nunique()),
                    "train_candidates": len(train),
                    "test_candidates": len(test),
                    "train_positives": int(y[train].sum()),
                    "test_positives": int(y[test].sum()),
                }
            )
            coefficients = model.named_steps["logisticregression"].coef_[0]
            for feature, coefficient in zip(features, coefficients):
                coefficient_rows.append(
                    {
                        "fold": fold,
                        "method": method,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                    }
                )
    return pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def disease_metrics(group: pd.DataFrame, score_column: str) -> dict:
    labels = group["label"].to_numpy(int)
    scores = group[score_column].to_numpy(float)
    tie_break = group["random"].to_numpy(float)
    order = np.lexsort((-tie_break, -scores))
    positive_count = int(labels.sum())
    k = min(10, len(group))
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


def paired_comparisons(
    by_disease: pd.DataFrame,
    comparison_methods: list[str],
    comparison_column: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    full = by_disease[by_disease["method"] == "raster_calibrated"].set_index("disease_id")
    paired_rows = []
    for metric in ALL_METRICS:
        metric_rows = []
        for method in comparison_methods:
            other = by_disease[by_disease["method"] == method].set_index("disease_id")
            difference = (full.loc[other.index, metric] - other[metric]).to_numpy(float)
            bootstrap = rng.choice(
                difference, size=(N_BOOTSTRAP, len(difference)), replace=True
            ).mean(axis=1)
            try:
                statistic, p_value = wilcoxon(
                    difference, alternative="greater", zero_method="wilcox"
                )
            except ValueError:
                statistic, p_value = np.nan, 1.0
            metric_rows.append(
                {
                    "metric": metric,
                    comparison_column: method,
                    "mean_difference": float(difference.mean()),
                    "ci_low": float(np.quantile(bootstrap, 0.025)),
                    "ci_high": float(np.quantile(bootstrap, 0.975)),
                    "win_rate": float((difference > 0).mean()),
                    "tie_rate": float((difference == 0).mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
        adjusted = bh_adjust(np.array([row["p_value"] for row in metric_rows]))
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["adjusted_p"] = float(adjusted_p)
            paired_rows.append(row)
    return pd.DataFrame(paired_rows)


def evaluate(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = [
        "random",
        "hpo_support",
        "target_popularity",
        "pharm_connectivity",
        "raster_heuristic",
        "raster_calibrated",
        "raster_no_hpo_local",
        "raster_no_pharmacology",
        "raster_no_ingredient_evidence",
    ]
    rows = []
    for disease_id, group in candidates.groupby("disease_id"):
        for method in methods:
            metrics = disease_metrics(group, method)
            rows.append(
                {
                    "disease_id": disease_id,
                    "disease_name": group["disease_name"].iloc[0],
                    "candidate_count": len(group),
                    "positive_count": int(group["label"].sum()),
                    "method": method,
                    **metrics,
                }
            )
    by_disease = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    summary_rows = []
    for method, group in by_disease.groupby("method"):
        for metric in ALL_METRICS:
            values = group[metric].to_numpy(float)
            samples = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "disease_count": len(values),
                }
            )
    summary = pd.DataFrame(summary_rows)

    comparison_methods = [
        "random",
        "hpo_support",
        "target_popularity",
        "pharm_connectivity",
        "raster_heuristic",
    ]
    paired = paired_comparisons(by_disease, comparison_methods, "baseline", rng)
    ablation_methods = [
        "raster_no_hpo_local",
        "raster_no_pharmacology",
        "raster_no_ingredient_evidence",
    ]
    ablations = paired_comparisons(by_disease, ablation_methods, "ablation", rng)
    return by_disease, summary, paired, ablations


def make_figure(
    candidates: pd.DataFrame,
    by_disease: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 9,
            "legend.frameon": False,
        }
    )
    methods = [
        "random",
        "hpo_support",
        "target_popularity",
        "pharm_connectivity",
        "raster_heuristic",
        "raster_calibrated",
    ]
    metrics = ["auprc", "ndcg10", "recall10", "mrr"]
    metric_labels = ["AUPRC", "nDCG@10", "Recall@10", "MRR"]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.55), gridspec_kw={"width_ratios": [1.65, 1.0, 1.0]})

    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.12
    for index, method in enumerate(methods):
        selected = summary[(summary["method"] == method)].set_index("metric").loc[metrics]
        means = selected["mean"].to_numpy()
        errors = np.vstack(
            [means - selected["ci_low"].to_numpy(), selected["ci_high"].to_numpy() - means]
        )
        ax.bar(
            x + (index - 2.5) * width,
            means,
            width,
            yerr=errors,
            color=COLORS[method],
            label=METHOD_LABELS[method],
            capsize=1.5,
            linewidth=0,
        )
    ax.set_xticks(x, metric_labels, fontsize=7)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Macro-average score")
    ax.set_title("A  Disease-held-out performance", loc="left")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    ax = axes[1]
    for method in methods:
        precision, recall, _ = precision_recall_curve(candidates["label"], candidates[method])
        ax.plot(recall, precision, color=COLORS[method], linewidth=1.3, label=METHOD_LABELS[method])
    ax.axhline(candidates["label"].mean(), color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("B  Pooled precision-recall", loc="left")
    ax.grid(color="#E6E6E6", linewidth=0.6)

    ax = axes[2]
    utility_metrics = ["precision10", "hit10"]
    utility_labels = ["Precision@10", "Queries with at least\none hit in top 10"]
    utility_methods = ["hpo_support", "raster_calibrated"]
    utility_colors = [COLORS[method] for method in utility_methods]
    x = np.arange(len(utility_metrics))
    width = 0.32
    for index, (method, color) in enumerate(zip(utility_methods, utility_colors)):
        selected = summary[summary["method"] == method].set_index("metric").loc[utility_metrics]
        means = 100 * selected["mean"].to_numpy()
        upper = 100 * selected["ci_high"].to_numpy()
        errors = 100 * np.vstack(
            [means / 100 - selected["ci_low"].to_numpy(), selected["ci_high"].to_numpy() - means / 100]
        )
        bars = ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=errors,
            color=color,
            capsize=2,
            linewidth=0,
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
    ax.set_ylim(0, 90)
    ax.set_ylabel("Percent (%)")
    ax.set_title("C  Top-10 screening utility", loc="left")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.29, wspace=0.38)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "cross_source_benchmark.svg", bbox_inches="tight")
    fig.savefig(output_dir / "cross_source_benchmark.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RASTER source-held-out ranking benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()

    candidates, provenance = build_candidates(args.project_root.resolve())
    folds, coefficients = fit_cross_validated_models(candidates)
    by_disease, summary, paired, ablations = evaluate(candidates)
    make_figure(candidates, by_disease, summary, args.figure_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(
        args.output_dir / "cross_source_candidate_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    folds.to_csv(args.output_dir / "cross_source_folds.csv", index=False)
    coefficients.to_csv(args.output_dir / "cross_source_coefficients.csv", index=False)
    by_disease.to_csv(args.output_dir / "cross_source_metrics_by_disease.csv", index=False)
    summary.to_csv(args.output_dir / "cross_source_method_summary.csv", index=False)
    paired.to_csv(args.output_dir / "cross_source_paired_comparisons.csv", index=False)
    ablations.to_csv(args.output_dir / "cross_source_ablation_comparisons.csv", index=False)

    dataset_summary = {
        "benchmark_definition": "Rank HPO disease-target candidates against source-held-out CTD direct marker/mechanism labels.",
        "ctd_excluded_from_features": True,
        "seed": SEED,
        "minimum_positives_per_disease": MIN_POSITIVES,
        "minimum_negatives_per_disease": MIN_NEGATIVES,
        "disease_count": int(candidates["disease_id"].nunique()),
        "candidate_count": len(candidates),
        "positive_count": int(candidates["label"].sum()),
        "positive_prevalence": float(candidates["label"].mean()),
        "pooled_auprc": {
            method: float(average_precision_score(candidates["label"], candidates[method]))
            for method in ["hpo_support", "raster_heuristic", "raster_calibrated"]
        },
        "pooled_auprc_enrichment_over_prevalence": {
            method: float(
                average_precision_score(candidates["label"], candidates[method])
                / candidates["label"].mean()
            )
            for method in ["hpo_support", "raster_heuristic", "raster_calibrated"]
        },
        "cross_validation": "Five-fold stratified group cross-validation with disease_id as the group.",
        "logistic_c": LOGISTIC_C,
        "bootstrap_replicates": N_BOOTSTRAP,
        "provenance": provenance,
    }
    (args.output_dir / "cross_source_benchmark_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print(summary[summary["method"].isin(["hpo_support", "raster_heuristic", "raster_calibrated"])].to_string(index=False))


if __name__ == "__main__":
    main()
