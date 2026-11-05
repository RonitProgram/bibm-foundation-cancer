from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NAMES = ["Logistic Regression", "Random Forest", "SGDClassifier", "Linear SVM"]
NON_RF_MODELS = ["Logistic Regression", "SGDClassifier", "Linear SVM"]
PLOT_METRICS = ["Recall", "Precision", "F1", "Balanced Accuracy", "ROC AUC", "PR AUC", "MCC"]
ALL_NUMERIC_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC AUC",
    "PR AUC",
    "Balanced Accuracy",
    "MCC",
    "TN",
    "FP",
    "FN",
    "TP",
]
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def representation_from_datatype(datatype):
    if datatype.startswith("bulkformer_"):
        return "BulkFormer"
    if datatype.startswith("bulkrnabert_"):
        return "BulkRNABERT"
    return "Other"


def source_from_filename(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return stem.replace("brcaresults_", "")


def parse_cv_rows_from_report(report_text, source):
    rows = []
    blocks = re.findall(
        r"Datatype:\s*(?P<dtype>[^\n]+)\n(?P<body>.*?)(?=\nDatatype:|\n={40,}|\Z)",
        report_text,
        flags=re.S,
    )

    for datatype, body in blocks:
        section_match = re.search(
            r"Cross-Validation Mean Metrics by Model\n(?P<section>.*?)(?=\n\n|\nHeld-Out Test Metrics|\n={40,}|\Z)",
            body,
            flags=re.S,
        )
        if not section_match:
            continue

        for raw_line in section_match.group("section").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Model"):
                continue

            model = None
            for name in MODEL_NAMES:
                if line.startswith(name):
                    model = name
                    break
            if model is None:
                continue

            tail = line[len(model) :].strip()
            nums = [float(x) for x in NUMBER_RE.findall(tail)]
            if len(nums) < len(ALL_NUMERIC_METRICS):
                continue

            nums = nums[: len(ALL_NUMERIC_METRICS)]
            row = {
                "Source": source,
                "Datatype": datatype,
                "Representation": representation_from_datatype(datatype),
                "Model": model,
            }
            for metric, value in zip(ALL_NUMERIC_METRICS, nums):
                row[metric] = value
            rows.append(row)

    return pd.DataFrame(rows)


def parse_classification_cv_means_from_ipynb(ipynb_text):
    metrics = ["ROC AUC", "Balanced Accuracy", "Recall", "Precision", "PR AUC", "F1", "MCC"]
    out = {}

    nb = json.loads(ipynb_text)
    stream_text_parts = []
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            text = output.get("text", "")
            stream_text_parts.append("".join(text) if isinstance(text, list) else str(text))

    combined_text = "\n".join(stream_text_parts)
    blocks = re.findall(
        r"===\s*(.*?)\s*===\n(.*?)(?=\n===\s*.*?\s*===|\Z)",
        combined_text,
        flags=re.S,
    )

    for model_name, block in blocks:
        if model_name not in MODEL_NAMES:
            continue

        mean_match = re.search(r"Mean metrics:\n(.*?)(?:\ndtype:\s*float64)", block, flags=re.S)
        if not mean_match:
            continue

        mean_text = mean_match.group(1)
        vals = {}
        for metric in metrics:
            m = re.search(rf"{re.escape(metric)}\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", mean_text)
            vals[metric] = float(m.group(1)) if m else np.nan

        out[model_name] = vals

    rows = []
    for model in MODEL_NAMES:
        vals = out.get(model, {})
        row = {
            "Source": "classification",
            "Datatype": "classification_ipynb",
            "Representation": "NotebookCV",
            "Model": model,
            "Accuracy": np.nan,
            "Specificity": np.nan,
            "TN": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "TP": np.nan,
        }
        for metric in ["Precision", "Recall", "F1", "ROC AUC", "PR AUC", "Balanced Accuracy", "MCC"]:
            row[metric] = vals.get(metric, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def assign_group(source):
    if source in {"fs", "classification"}:
        return "fs_and_classification"
    if source in {"hvg1k156", "hvg1k312", "hvg1000"}:
        return "hvg1k_and_hvg1000"
    if source in {"hvg10k156", "hvg10k312", "hvg10000"}:
        return "hvg10k_and_hvg10000"
    if source in {"litgenes", "litgenes2"}:
        return "litgenes"
    return "other"


def plot_metrics_by_source_model(df_agg, representation, model_order, out_path, title):
    df = df_agg[
        (df_agg["Representation"] == representation)
        & df_agg["Model"].isin(model_order)
    ].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for representation: {representation}")

    sources = list(dict.fromkeys(df["Source"].tolist()))
    metrics = PLOT_METRICS
    width = 0.18 if len(model_order) == 4 else 0.24

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    axes = axes.flatten()

    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#ff7f0e",
        "SGDClassifier": "#2ca02c",
        "Linear SVM": "#d62728",
    }

    x = np.arange(len(sources))
    offsets = np.linspace(-width * (len(model_order) - 1) / 2.0, width * (len(model_order) - 1) / 2.0, len(model_order))

    for ax, metric in zip(axes, metrics):
        all_vals = []
        for idx, model in enumerate(model_order):
            vals = []
            for source in sources:
                row = df[(df["Source"] == source) & (df["Model"] == model)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + offsets[idx], vals, width=width, label=model, color=colors.get(model))
            all_vals.extend([v for v in vals if not np.isnan(v)])

        y_min = min(all_vals) if all_vals else 0.0
        ax.set_ylim(min(-0.15, y_min - 0.05), 1.05)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=25, ha="right")

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=False)
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create grouped BRCA result outputs and plots from brcaresults + classification.ipynb."
    )
    parser.add_argument(
        "--brcaresults-dir",
        default=str(ROOT / "results/raw_reports"),
        help="Directory containing brcaresults_*.txt files",
    )
    parser.add_argument(
        "--classification-ipynb",
        default=str(ROOT / "notebooks/classification.ipynb"),
        help="Path to classification.ipynb",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "results/rebuilt"),
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_dir = os.path.join(args.outdir, "raw_brcaresults")
    grouped_dir = os.path.join(args.outdir, "groups")
    plots_dir = os.path.join(args.outdir, "plots")
    tables_dir = os.path.join(args.outdir, "tables")

    for d in [raw_dir, grouped_dir, plots_dir, tables_dir]:
        os.makedirs(d, exist_ok=True)

    result_paths = sorted(glob(os.path.join(args.brcaresults_dir, "brcaresults_*.txt")))
    if not result_paths:
        raise RuntimeError(f"No brcaresults_*.txt files found in: {args.brcaresults_dir}")

    all_rows = []
    source_to_file = {}

    for path in result_paths:
        source = source_from_filename(path)
        source_to_file[source] = path

        shutil.copy2(path, os.path.join(raw_dir, os.path.basename(path)))

        with open(path, "r", encoding="utf-8") as f:
            report_text = f.read()

        df_source = parse_cv_rows_from_report(report_text, source)
        if not df_source.empty:
            all_rows.append(df_source)

    with open(args.classification_ipynb, "r", encoding="utf-8") as f:
        class_text = f.read()
    df_class = parse_classification_cv_means_from_ipynb(class_text)
    all_rows.append(df_class)

    df_long = pd.concat(all_rows, ignore_index=True)
    if df_long.empty:
        raise RuntimeError("No CV metrics could be parsed from brcaresults or classification notebook.")

    numeric_cols = [c for c in ALL_NUMERIC_METRICS if c in df_long.columns]
    df_agg = (
        df_long.groupby(["Source", "Representation", "Model"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
        .sort_values(["Source", "Model"])
        .reset_index(drop=True)
    )
    df_agg["Group"] = df_agg["Source"].apply(assign_group)

    # Keep a stable source order so plots are easier to scan.
    preferred_order = [
        "fs",
        "classification",
        "hvg1k156",
        "hvg1k312",
        "hvg1000",
        "hvg10k156",
        "hvg10k312",
        "hvg10000",
        "litgenes",
        "litgenes2",
    ]
    rank = {name: idx for idx, name in enumerate(preferred_order)}
    df_agg["_rank"] = df_agg["Source"].map(rank).fillna(999).astype(int)
    df_agg = df_agg.sort_values(["_rank", "Model"]).drop(columns=["_rank"]).reset_index(drop=True)

    long_csv = os.path.join(tables_dir, "cv_metrics_long_by_datatype_model.csv")
    agg_csv = os.path.join(tables_dir, "cv_metrics_mean_by_source_model.csv")
    df_long.to_csv(long_csv, index=False)
    df_agg.to_csv(agg_csv, index=False)

    groups = sorted(df_agg["Group"].dropna().unique().tolist())
    for group in groups:
        group_path = os.path.join(grouped_dir, group)
        os.makedirs(group_path, exist_ok=True)

        df_group = df_agg[df_agg["Group"] == group].copy()
        df_group.to_csv(os.path.join(group_path, f"cv_metrics_{group}.csv"), index=False)

        sources = sorted(df_group["Source"].unique().tolist())
        for source in sources:
            if source in source_to_file:
                shutil.copy2(source_to_file[source], os.path.join(group_path, os.path.basename(source_to_file[source])))

        if group == "fs_and_classification":
            shutil.copy2(args.classification_ipynb, os.path.join(group_path, "classification.ipynb"))

    plot_paths = []
    for representation in ["BulkFormer", "BulkRNABERT"]:
        representation_slug = representation.lower()
        with_rf_png = os.path.join(
            plots_dir,
            f"{representation_slug}_cv_comparison_with_rf.png",
        )
        no_rf_png = os.path.join(
            plots_dir,
            f"{representation_slug}_cv_comparison_without_rf.png",
        )

        plot_metrics_by_source_model(
            df_agg=df_agg,
            representation=representation,
            model_order=MODEL_NAMES,
            out_path=with_rf_png,
            title=f"{representation} 5-Fold CV Mean Metrics (With Random Forest)",
        )
        plot_metrics_by_source_model(
            df_agg=df_agg,
            representation=representation,
            model_order=NON_RF_MODELS,
            out_path=no_rf_png,
            title=f"{representation} 5-Fold CV Mean Metrics (Without Random Forest)",
        )
        plot_paths.extend([with_rf_png, no_rf_png])

    print("Saved outputs:")
    print(f"- {raw_dir}")
    print(f"- {grouped_dir}")
    print(f"- {long_csv}")
    print(f"- {agg_csv}")
    for plot_path in plot_paths:
        print(f"- {plot_path}")


if __name__ == "__main__":
    main()
