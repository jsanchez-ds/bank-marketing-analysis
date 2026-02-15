"""
Utility functions for Bank Marketing Analysis project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)


def load_data(filepath: str = "data/bank-additional-full.csv") -> pd.DataFrame:
    """Load and perform initial cleaning of the bank marketing dataset."""
    df = pd.read_csv(filepath, sep=";")
    # Replace 'unknown' with NaN for cleaner analysis
    df.replace("unknown", np.nan, inplace=True)
    return df


def plot_target_distribution(df: pd.DataFrame, target_col: str = "y") -> None:
    """Plot the distribution of the target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Count plot
    df[target_col].value_counts().plot(kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Target Distribution (Count)")
    axes[0].set_ylabel("Count")
    
    # Percentage plot
    df[target_col].value_counts(normalize=True).plot(
        kind="pie", ax=axes[1], autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"]
    )
    axes[1].set_title("Target Distribution (%)")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    plt.savefig("figures/target_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_conversion_by_feature(df: pd.DataFrame, feature: str, target: str = "y") -> None:
    """Plot conversion rate by a categorical feature."""
    conv_rate = df.groupby(feature)[target].apply(lambda x: (x == "yes").mean()).sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    conv_rate.plot(kind="barh", ax=ax, color="#3498db")
    ax.set_xlabel("Conversion Rate")
    ax.set_title(f"Conversion Rate by {feature}")
    ax.axvline(x=df[target].apply(lambda x: x == "yes").mean(), color="red", linestyle="--", label="Overall avg")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"figures/conversion_by_{feature}.png", dpi=150, bbox_inches="tight")
    plt.show()


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Evaluate a classification model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1),
        "Recall": recall_score(y_test, y_pred, pos_label=1),
        "F1-Score": f1_score(y_test, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))
    
    return metrics


def plot_roc_curves(models_dict: dict, X_test, y_test) -> None:
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("figures/roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name: str = "Model") -> None:
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    
    plt.tight_layout()
    plt.savefig(f"figures/cm_{model_name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.show()
