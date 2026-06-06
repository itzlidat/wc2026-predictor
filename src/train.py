import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from elo import DATA_DIR
from features import FEATURE_COLS

PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"
MODEL_PATH = DATA_DIR / "model.json"

TARGET_COL = "target"
LABEL_NAMES = {0: "away_win", 1: "draw", 2: "home_win"}


DRAW_WEIGHT = 1.5  # draws oversampled relative to wins/losses
WC_WEIGHT   = 2.0  # FIFA World Cup matches upweighted vs all other competitions


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["date"])
    return df[FEATURE_COLS], df[TARGET_COL], df["is_world_cup"]


def make_sample_weights(y: pd.Series, is_wc: pd.Series) -> np.ndarray:
    """
    Combine draw and World Cup multipliers multiplicatively:
      regular non-draw : 1.0
      regular draw     : 1.5
      WC non-draw      : 2.0
      WC draw          : 3.0
    """
    draw_w = np.where(y == 1, DRAW_WEIGHT, 1.0)
    wc_w   = np.where(is_wc == 1, WC_WEIGHT, 1.0)
    return draw_w * wc_w


def train(X: pd.DataFrame, y: pd.Series, is_wc: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=make_sample_weights(y, is_wc))
    return model


def plot_feature_importance(model: XGBClassifier) -> None:
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance (F-score)")
    ax.set_title("XGBoost Feature Importance — WC2026 Predictor")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = PREDICTIONS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Feature importance chart saved to {out}")


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print()
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        )
    )


if __name__ == "__main__":
    X, y, is_wc = load_data()

    X_train, X_test, y_train, y_test, wc_train, wc_test = train_test_split(
        X, y, is_wc, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"WC matches in train: {wc_train.sum():,}  |  test: {wc_test.sum():,}")
    print(f"Class distribution (train): {y_train.value_counts().sort_index().to_dict()}\n")

    print("Training XGBoost...")
    model = train(X_train, y_train, wc_train)

    print("\n--- Evaluation ---")
    evaluate(model, X_test, y_test)

    plot_feature_importance(model)

    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
