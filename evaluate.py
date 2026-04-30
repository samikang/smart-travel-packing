#!/usr/bin/env python3
"""
Pillar 1 Evaluation – Multi‑Label Recommender Accuracy
=======================================================
Usage:
    python evaluate.py --model knn      # evaluate one model
    python evaluate.py --compare        # compare all trainable models
    python evaluate.py --model lgbm --cv 5  # 5‑fold cross‑validation
"""

import argparse
import numpy as np
from recommender import (
    _generate_training_data,
    _load_or_train,
    ALL_ITEMS,
    VALID_MODEL_TYPES,
)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ────────────────────────────────────────────────────────────────────────────
def get_data(test_size=0.2, seed=42):
    """Generate synthetic data and split into train/test."""
    X, Y = _generate_training_data(n_samples=15_000, seed=seed)
    from sklearn.model_selection import train_test_split
    return train_test_split(X, Y, test_size=test_size, random_state=seed)


def compute_metrics(model, X_test, Y_test):
    """Calculate per‑item and overall metrics."""
    Y_pred = model.predict(X_test)
    metrics = {}
    for i, item in enumerate(ALL_ITEMS):
        y_true = Y_test[:, i]
        y_pred = Y_pred[:, i]
        metrics[item] = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
        }
    overall = {
        'macro_avg_precision': precision_score(Y_test, Y_pred, average='macro', zero_division=0),
        'macro_avg_recall':    recall_score(Y_test, Y_pred, average='macro', zero_division=0),
        'macro_avg_f1':        f1_score(Y_test, Y_pred, average='macro', zero_division=0),
        'exact_match_ratio':   np.mean(np.all(Y_test == Y_pred, axis=1)),
    }
    return metrics, overall


def print_single_result(metrics, overall, model_name):
    """Print a nicely formatted table for a single model."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title=f"Evaluation for {model_name}", show_lines=True)
        table.add_column("Item", style="cyan", no_wrap=True, width=30)
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")
        for item, m in metrics.items():
            table.add_row(
                item,
                f"{m['accuracy']:.3f}",
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1']:.3f}",
            )
        console.print(table)
        console.print(f"\nOverall metrics:")
        console.print(f"  Macro avg precision: {overall['macro_avg_precision']:.3f}")
        console.print(f"  Macro avg recall:    {overall['macro_avg_recall']:.3f}")
        console.print(f"  Macro avg F1:         {overall['macro_avg_f1']:.3f}")
        console.print(f"  Exact match ratio:    {overall['exact_match_ratio']:.3f}")
    except ImportError:
        print(f"\n{'─'*60}")
        print(f"Evaluation for {model_name}")
        print(f"{'─'*60}")
        for item, m in metrics.items():
            print(f"{item:<30} {m['accuracy']:.3f}  {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}")
        print(f"{'─'*60}")
        print(f"Overall:")
        print(f"  Macro avg precision: {overall['macro_avg_precision']:.3f}")
        print(f"  Macro avg recall:    {overall['macro_avg_recall']:.3f}")
        print(f"  Macro avg F1:         {overall['macro_avg_f1']:.3f}")
        print(f"  Exact match ratio:    {overall['exact_match_ratio']:.3f}")


def evaluate_single(model_type):
    """Evaluate one model and print results."""
    print(f"[Evaluator] Training and evaluating '{model_type}' ...")
    X_train, X_test, Y_train, Y_test = get_data()
    model = _load_or_train(model_type)
    # If model is MultiOutputClassifier, we need to train it explicitly because _load_or_train already trains if not cached.
    # But _load_or_train returns a fitted model.
    metrics, overall = compute_metrics(model, X_test, Y_test)
    print_single_result(metrics, overall, model_type)


def evaluate_compare():
    """Compare all trainable models and show a summary."""
    print("[Evaluator] Comparing all models ...")
    X_train, X_test, Y_train, Y_test = get_data()
    results = []
    for model_type in ["knn", "lgbm", "random_forest"]:
        print(f"  Working on {model_type} ...")
        model = _load_or_train(model_type)
        _, overall = compute_metrics(model, X_test, Y_test)
        results.append((model_type, overall))
    # Print comparison table
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Model Comparison (Macro avg)")
        table.add_column("Model", style="bold cyan")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Exact Match", justify="right")
        for model_type, overall in results:
            table.add_row(
                model_type,
                f"{overall['macro_avg_precision']:.3f}",
                f"{overall['macro_avg_recall']:.3f}",
                f"{overall['macro_avg_f1']:.3f}",
                f"{overall['exact_match_ratio']:.3f}",
            )
        console.print(table)
    except ImportError:
        print(f"\n{'Model':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Exact Match':>12}")
        for model_type, overall in results:
            print(f"{model_type:<15} {overall['macro_avg_precision']:>10.3f} "
                  f"{overall['macro_avg_recall']:>10.3f} {overall['macro_avg_f1']:>10.3f} "
                  f"{overall['exact_match_ratio']:>12.3f}")


def evaluate_cv(model_type, folds=5):
    """Run cross‑validation on a single model."""
    print(f"[Evaluator] {folds}‑fold cross‑validation for '{model_type}' ...")
    X, Y = _generate_training_data(n_samples=15_000, seed=42)
    # For cross‑validation we need a classifier that handles multi‑output directly.
    # We'll use the same MultiOutputClassifier wrapper but fit on each fold.
    from sklearn.multioutput import MultiOutputClassifier
    if model_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        base = KNeighborsClassifier(n_neighbors=15, weights="distance", metric="euclidean")
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier
        base = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20, verbose=-1)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        base = RandomForestClassifier(n_estimators=200, max_depth=12,
                                      min_samples_leaf=10, n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    clf = MultiOutputClassifier(base, n_jobs=-1)
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(clf, X, Y, cv=folds, scoring=scoring, n_jobs=-1)
    print(f"Cross‑validation results ({folds} folds):")
    for metric in scoring:
        vals = scores[f'test_{metric}']
        print(f"  {metric}: {vals.mean():.3f} ± {vals.std():.3f}")


# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Packing Recommender Model Evaluation")
    parser.add_argument("--model", default=None, choices=["knn", "lgbm", "random_forest"],
                        help="Model to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all trainable models")
    parser.add_argument("--cv", type=int, default=0,
                        help="Number of cross‑validation folds (requires --model)")
    args = parser.parse_args()

    if args.compare:
        evaluate_compare()
        return
    if args.cv:
        if not args.model:
            raise SystemExit("Please specify --model when using --cv")
        evaluate_cv(args.model, args.cv)
        return
    if args.model:
        evaluate_single(args.model)
        return
    parser.print_help()


if __name__ == "__main__":
    main()