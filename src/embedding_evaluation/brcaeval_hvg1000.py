import gc
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pipeline_paths import embedding_input
parser = argparse.ArgumentParser(description='Evaluate BRCA embeddings using the archived evaluation algorithm.')
parser.add_argument('--embeddings-source', choices=['archived', 'generated'], default='archived')
parser.add_argument('--output', type=Path)
parser.add_argument('--top-k', type=int, choices=[0,156,312], default=312, help='0 keeps the uncapped frequency-selected coordinates')
args = parser.parse_args()
BASE_DIR = str(ROOT / 'data/expression')
suffix = str(args.top_k) if args.top_k else ''
report_name = 'brcaresults_hvg1k' + suffix + '.txt' if args.top_k else 'brcaresults_hvg1000.txt'
DATASET_PATHS = {
    'bulkformer_hvg1000': str(embedding_input('bf', 'hvg1000.npy', args.embeddings_source)),
    'bulkrnabert_hvg1000': str(embedding_input('bert', 'hvg1000bert.npy', args.embeddings_source)),
}
output = args.output or ROOT / 'results/generated_reports' / args.embeddings_source / report_name
output.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = str(output)

MODEL_FACTORIES = {
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(copy=False),
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    ),
    "SGDClassifier": lambda: make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            class_weight="balanced",
            alpha=0.0001,
            l1_ratio=0.15,
            max_iter=1000,
            random_state=42,
        ),
    ),
    "Linear SVM": lambda: make_pipeline(
        StandardScaler(),
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000,
        ),
    ),
}


def _to_1d(y):
    if isinstance(y, (pd.DataFrame, pd.Series)):
        return y.squeeze().to_numpy()
    return np.asarray(y).ravel()


def to_2d_features(X):
    X = np.asarray(X)
    if X.ndim == 2:
        return X
    if X.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {X.shape}")
    return X.reshape(X.shape[0], -1)


def apply_feature_selection(X_train, y_train, X_test):
    # Match classification.ipynb: ElasticNet trials + non-zero coefficient frequency threshold.
    alpha = 0.001
    l1_ratio = 0.5
    n_trials = 100
    threshold = 0.8
    top_k_genes = args.top_k

    y_train_float = np.asarray(y_train, dtype=np.float32)
    coefficients_index = np.zeros(X_train.shape[1], dtype=np.float32)

    for i in range(n_trials):
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=i,
            max_iter=10000,
        )
        model.fit(X_train, y_train_float)
        nonzero_indices = np.where(np.abs(model.coef_) > 1e-8)[0]
        coefficients_index[nonzero_indices] += 1.0

    selected_feature_index = np.where(coefficients_index >= (threshold * n_trials))[0]

    # Safety fallback for degenerate cases.
    if selected_feature_index.size == 0:
        selected_feature_index = np.arange(X_train.shape[1])

    # Keep only the top-k features among the selected set.
    selected_scores = coefficients_index[selected_feature_index]
    rank_order = np.lexsort((selected_feature_index, -selected_scores))
    selected_feature_index = selected_feature_index[rank_order[: min(top_k_genes, selected_feature_index.size) if top_k_genes else selected_feature_index.size]]

    X_train_fs = X_train[:, selected_feature_index]
    X_test_fs = X_test[:, selected_feature_index]

    print(
        f"Feature selection (classification style, top {top_k_genes}): {X_train.shape[1]} -> {X_train_fs.shape[1]}",
        flush=True,
    )

    return X_train_fs.astype(np.float32), X_test_fs.astype(np.float32)


def evaluate_model(model, X_test, y_test):
    y_test_1d = _to_1d(y_test)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test_1d, y_pred).ravel()
    specificity = tn / (tn + fp)

    return {
        "Accuracy": accuracy_score(y_test_1d, y_pred),
        "Precision": precision_score(y_test_1d, y_pred, zero_division=0),
        "Recall": recall_score(y_test_1d, y_pred),
        "Specificity": specificity,
        "F1": f1_score(y_test_1d, y_pred),
        "ROC AUC": roc_auc_score(y_test_1d, y_score),
        "PR AUC": average_precision_score(y_test_1d, y_score),
        "Balanced Accuracy": balanced_accuracy_score(y_test_1d, y_pred),
        "MCC": matthews_corrcoef(y_test_1d, y_pred),
        "Confusion Matrix": confusion_matrix(y_test_1d, y_pred, labels=[0, 1]),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def cross_validate_model(model_factory, X_data, y_data, cv):
    fold_results = []
    y_data_1d = _to_1d(y_data)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data_1d), start=1):
        X_fold_train = np.asarray(X_data[train_idx], dtype=np.float32)
        X_fold_val = np.asarray(X_data[val_idx], dtype=np.float32)
        y_fold_train = y_data_1d[train_idx]
        y_fold_val = y_data_1d[val_idx]

        model = model_factory()
        model.fit(X_fold_train, y_fold_train)

        fold_result = evaluate_model(model, X_fold_val, y_fold_val)
        fold_result["Fold"] = fold
        fold_results.append(fold_result)

        del model, X_fold_train, X_fold_val
        gc.collect()

    return pd.DataFrame(fold_results)


def write_report(all_results_by_datatype, all_cv_means, all_test_results):
    with open(OUTPUT_FILE, "w") as f:
        f.write("BRCA HVG1000 Evaluation Results\n")
        f.write("=" * 80 + "\n\n")

        for datatype, payload in all_results_by_datatype.items():
            f.write(f"Datatype: {datatype}\n")
            f.write("-" * 80 + "\n")

            f.write("Cross-Validation Mean Metrics by Model\n")
            cv_mean = (
                payload["cv_summary"]
                .drop(columns=["Fold", "Confusion Matrix"])
                .groupby("Model", as_index=False)
                .mean(numeric_only=True)
            )
            with pd.option_context("display.max_columns", None, "display.width", 200):
                f.write(cv_mean.to_string(index=False))
            f.write("\n\n")

            f.write("Held-Out Test Metrics by Model\n")
            test_summary = payload["test_summary"].copy()
            with pd.option_context("display.max_columns", None, "display.width", 200):
                f.write(test_summary.to_string())
            f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("Combined Tables\n")
        f.write("=" * 80 + "\n\n")

        f.write("All CV Means (Datatype + Model)\n")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            f.write(all_cv_means.to_string(index=False))
        f.write("\n\n")

        f.write("All Test Results (Datatype + Model)\n")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            f.write(all_test_results.to_string(index=False))
        f.write("\n")


if __name__ == "__main__":
    for array_path in DATASET_PATHS.values():
        if not Path(array_path).is_file():
            raise FileNotFoundError(f'Missing embedding array: {array_path}. Check provenance/path_audit.json.')
    y = pd.read_csv(os.path.join(BASE_DIR, "yiv.csv")).iloc[:, 0].to_numpy().ravel()
    all_indices = np.arange(len(y))

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results_by_datatype = {}

    for datatype, array_path in DATASET_PATHS.items():
        print(f"\nLoading {datatype}", flush=True)
        X = np.load(array_path, mmap_mode="r")

        original_shape = X.shape
        X = to_2d_features(X)

        if X.shape[0] != len(y):
            raise ValueError(f"{datatype}: {X.shape[0]} samples, but y has {len(y)} labels")

        print(f"{datatype}: original_shape={original_shape} -> model_shape={X.shape}", flush=True)

        X_train = np.asarray(X[train_indices], dtype=np.float32)
        X_test = np.asarray(X[test_indices], dtype=np.float32)
        y_train = y[train_indices]
        y_test = y[test_indices]

        del X
        gc.collect()

        X_train, X_test = apply_feature_selection(X_train, y_train, X_test)

        cv_results = {}
        test_results = {}

        for name, factory in MODEL_FACTORIES.items():
            print(f"Training {datatype}: {name}", flush=True)

            fold_results = cross_validate_model(factory, X_train, y_train, cv)
            cv_results[name] = fold_results

            final_model = factory()
            final_model.fit(X_train, y_train)
            test_results[name] = evaluate_model(final_model, X_test, y_test)

            del final_model
            gc.collect()

        cv_summary = pd.concat(
            [result.assign(Model=name) for name, result in cv_results.items()],
            ignore_index=True,
        )
        test_summary = pd.DataFrame(test_results).T

        all_results_by_datatype[datatype] = {
            "cv_results": cv_results,
            "cv_summary": cv_summary,
            "test_results": test_results,
            "test_summary": test_summary,
        }

        del X_train, X_test, y_train, y_test
        gc.collect()

    mean_cv_tables = []
    for datatype, payload in all_results_by_datatype.items():
        cv_mean = (
            payload["cv_summary"]
            .drop(columns=["Fold", "Confusion Matrix"])
            .groupby("Model", as_index=False)
            .mean(numeric_only=True)
        )
        cv_mean.insert(0, "Datatype", datatype)
        mean_cv_tables.append(cv_mean)

    all_cv_means = pd.concat(mean_cv_tables, ignore_index=True)

    test_tables = []
    for datatype, payload in all_results_by_datatype.items():
        t = payload["test_summary"].reset_index().rename(columns={"index": "Model"})
        t.insert(0, "Datatype", datatype)
        test_tables.append(t)

    all_test_results = pd.concat(test_tables, ignore_index=True)

    write_report(all_results_by_datatype, all_cv_means, all_test_results)

    print("\nSaved outputs:")
    print("- all_results_by_datatype")
    print("- all_cv_means")
    print("- all_test_results")
    print(f"- report: {OUTPUT_FILE}")
