"""BRCA-only extraction of plot_combined_results_blresults.py; plotting logic preserved."""
from pathlib import Path
import string
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
BREAST_CV_PATH = ROOT / "results/tables/cv_metrics_long_by_datatype_model.csv"
OUTPUT_DIR = ROOT / "figures/regenerated"
def strategy_key_labels(disease, foundation_model, feature_sets):
    return [label for _, label in feature_sets]
MODELS = ["Logistic Regression", "SGDClassifier", "Linear SVM"]

METRICS = [
    "Precision",
    "Recall",
    "F1",
    "ROC AUC",
    "PR AUC",
    "Balanced Accuracy",
    "MCC",
]

BREAST_FEATURE_SETS = [
    ("classification", "Standard FS"),
    ("hvg1000", "HVG1k → 64"),
    ("hvg1k156", "HVG1k → 156"),
    ("hvg1k312", "HVG1k → 312"),
    ("hvg10000", "HVG10k → 153"),
    ("hvg10k156", "HVG10k → 156"),
    ("hvg10k312", "HVG10k → 312"),
    ("litgenes", "LitGenes-GSEA"),
    ("litgenes2", "LitGenes-MetaSig"),
]

COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest": "#ff7f0e",
    "SGDClassifier": "#2ca02c",
    "Linear SVM": "#d62728",
}

PANEL_LETTERS = ["A", "B"]

FEATURE_SET_LETTERS = list(string.ascii_lowercase)

TITLE_FONTSIZE = 26

AXIS_TITLE_FONTSIZE = 36

TICK_FONTSIZE = 22

LEGEND_FONTSIZE = 34

PANEL_LABEL_FONTSIZE = 42

CAPTION_FONTSIZE = 20

def load_breast_metrics() -> pd.DataFrame:
    data = pd.read_csv(BREAST_CV_PATH)
    selected_sources = [source for source, _ in BREAST_FEATURE_SETS]
    data = data[data["Source"].isin(selected_sources) & data["Model"].isin(MODELS)].copy()
    label_map = dict(BREAST_FEATURE_SETS)
    data["Feature Set"] = data["Source"].map(label_map)

    classification = data[data["Source"] == "classification"].copy()
    classification["Foundation Model"] = "Classification baseline"
    embeddings = data[data["Source"] != "classification"].copy()
    embeddings["Foundation Model"] = embeddings["Representation"]
    return pd.concat([classification, embeddings], ignore_index=True)

def plot_panel(axes_block, data: pd.DataFrame, foundation_model: str, feature_sets: list, feature_letters: list) -> list:
    """Draw the 2x4 metric grid for one foundation model into the given axes."""
    view = data[data["Foundation Model"] == foundation_model].copy()
    baseline = data[data["Foundation Model"] == "Classification baseline"].copy()
    view = pd.concat([baseline, view], ignore_index=True)
    labels = [label for _, label in feature_sets]
    x = np.arange(len(labels)) * 1.2
    width = 0.25
    offsets = np.arange(len(MODELS)) * width - width

    axes_flat = axes_block.flatten()
    handles, legend_labels = [], []
    for axis, metric in zip(axes_flat, METRICS):
        values = []
        for offset, model in zip(offsets, MODELS):
            model_values = []
            for label in labels:
                row = view[(view["Feature Set"] == label) & (view["Model"] == model)]
                value = float(row[metric].iloc[0]) if not row.empty else np.nan
                model_values.append(value)
                if not np.isnan(value):
                    values.append(value)
            bars = axis.bar(x + offset, model_values, width, label=model, color=COLORS[model])
            if model not in legend_labels:
                handles.append(bars)
                legend_labels.append(model)

        axis.set_title(metric, fontsize=AXIS_TITLE_FONTSIZE)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x)
        axis.set_xticklabels(
            feature_letters,
            fontsize=TICK_FONTSIZE,
            fontweight="bold",
            ha="center",
        )
        axis.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        axis.set_ylim(min(-0.15, min(values) - 0.05) if values else -0.15, 1.05)

    for axis in axes_flat[len(METRICS):]:
        axis.set_visible(False)

    return handles, legend_labels

def save_combined_figure(data: pd.DataFrame, disease: str, feature_sets: list, filename: str) -> None:
    foundation_models = ["BulkFormer", "BulkRNABERT"]
    labels = [label for _, label in feature_sets]

    fig, axes = plt.subplots(4, 4, figsize=(42, 24))

    all_handles, all_legend_labels = [], []
    panel_keys = []
    for panel_index, foundation_model in enumerate(foundation_models):
        row_start = panel_index * 2
        panel_axes = axes[row_start:row_start + 2, :]
        feature_letters = list(string.ascii_lowercase[: len(labels)])
        handles, legend_labels = plot_panel(panel_axes, data, foundation_model, feature_sets, feature_letters)
        panel_keys.append(
            (
                PANEL_LETTERS[panel_index],
                foundation_model,
                feature_letters,
                strategy_key_labels(disease, foundation_model, feature_sets),
            )
        )
        for handle, label in zip(handles, legend_labels):
            if label not in all_legend_labels:
                all_handles.append(handle)
                all_legend_labels.append(label)

        panel_letter = PANEL_LETTERS[panel_index]
        fig.text(
            0.005,
            0.69 if panel_index == 0 else 0.27,
            f"Panel {panel_letter} \u2014 {foundation_model}",
            fontsize=PANEL_LABEL_FONTSIZE,
            fontweight="bold",
            va="center",
            ha="left",
            rotation=90,
        )

    legend_axis = axes[3, 3]
    legend_axis.set_visible(True)
    legend_axis.axis("off")
    legend_axis.legend(
        all_handles,
        all_legend_labels,
        loc="center",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=1,
    )

    fig.tight_layout(rect=[0.035, 0.015, 1, 0.99], h_pad=0.8, w_pad=0.8)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)

    key_fig, key_axis = plt.subplots(figsize=(8.5, 24))
    key_axis.axis("off")
    key_axis.set_position([0, 0, 1, 1])
    key_line_spacing = 0.046
    key_panel_gap = 0.02
    key_text_height = 27 / (24 * 72)
    key_text_advance = sum(
        key_line_spacing * (1 + len(panel_labels)) for _, _, _, panel_labels in panel_keys
    ) + key_panel_gap * max(0, len(panel_keys) - 1)
    y_position = (1 + key_text_advance + key_text_height) / 2
    for panel_index, (panel_letter, foundation_model, feature_letters, panel_labels) in enumerate(panel_keys):
        key_axis.text(0.04, y_position, f"Panel {panel_letter} - {foundation_model}",
                      fontsize=36, fontweight="bold", ha="left", va="top")
        y_position -= key_line_spacing
        for letter, label in zip(feature_letters, panel_labels):
            key_axis.text(0.08, y_position, f"{letter}  =  {label}", fontsize=27,
                          ha="left", va="top")
            y_position -= key_line_spacing
        if panel_index < len(panel_keys) - 1:
            y_position -= key_panel_gap
    key_filename = Path(filename).with_name(Path(filename).stem + "_feature_selection_key.png")
    key_fig.savefig(OUTPUT_DIR / key_filename, dpi=220)
    plt.close(key_fig)
if __name__ == "__main__":
    plt.rcParams.update({"font.size": TICK_FONTSIZE})
    save_combined_figure(load_breast_metrics(), "Breast", BREAST_FEATURE_SETS,
                         "figure2_breast_combined_cross_validation_metrics.png")
    print(f"Saved BRCA figure and key to {OUTPUT_DIR}")
