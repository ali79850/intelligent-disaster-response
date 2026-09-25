"""
Phase 4, Baseline 2: Train a random forest classifier on hand-crafted
per-building features, evaluate honestly on train/val (sanity check) and
test (final, reported result).
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    "mean_diff", "std_diff", "mean_pre", "mean_post", "area_pixels",
    "edge_density_pre", "edge_density_post", "edge_density_change",
]

def load_split(name):
    df = pd.read_csv(f"data/processed/features/{name}_features.csv")
    X = df[FEATURE_COLS].values
    y = df["subtype"].values
    return X, y, df

def main():
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, _ = load_split("test")

    print(f"Train: {len(y_train)} buildings, Val: {len(y_val)}, Test: {len(y_test)}")

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("\nTraining random forest...")
    clf.fit(X_train, y_train)

    print("\n=== VAL SET (sanity check during development) ===")
    val_pred = clf.predict(X_val)
    print(classification_report(y_val, val_pred, zero_division=0))

    print("\n=== TEST SET (final, honest result) ===")
    test_pred = clf.predict(X_test)
    print(classification_report(y_test, test_pred, zero_division=0))

    print("\n=== Test set confusion matrix ===")
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    print("Labels:", labels)
    print(cm)

    print("\n=== Feature importances ===")
    for name, importance in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s} {importance:.4f}")

if __name__ == "__main__":
    main()