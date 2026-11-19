# RASTER-Agent manuscript analysis

This repository contains the archived Python analysis scripts and aggregate benchmark results for the RASTER-Agent manuscript. It does not contain the full application, third-party input databases, candidate-level records, or the language-model evaluation dataset. The repository alone is not sufficient to rerun the analyses from raw inputs.

- `analysis/cross_source_benchmark.py`: disease–target benchmark.
- `analysis/compound_target_benchmark.py`: compound–target benchmark.
- `analysis/case_robustness.py`: sensitivity analysis of the case-study rankings.
- `results/`: macro-averaged metric summaries, paired baseline comparisons, and disease–target ablation comparisons.

The benchmark scripts require NumPy, pandas, SciPy, scikit-learn, and Matplotlib. They accept `--project-root` pointing to a project tree with the original input files. The case-study script requires the archived case outputs; use `--help` to view its inputs.

The aggregate results can be inspected without the original databases. Reported macro-average AUPRC is 0.2953 for disease–target prioritization and 0.3294 for compound–target prioritization.
