import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from dataset_generator import generate, FEATURE_COLS, LIFE_EVENTS

MODEL_DIR   = "model"
MODEL_PATH  = f"{MODEL_DIR}/lsti_rf_model.pkl"
ENCODER_PATH= f"{MODEL_DIR}/label_encoder.pkl"
META_PATH   = f"{MODEL_DIR}/model_meta.json"

def train():
    print("=" * 60)
    print("  LifeSignal AI — LSTI Model Training")
    print("=" * 60)

    print("\n[1/5] Generating dataset...")
    df = generate()

    print("\n[2/5] Preparing features...")
    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df["life_event_label"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    print("\n[3/5] Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=35,
        max_depth=3,
        min_samples_split=250,
        min_samples_leaf=130,
        max_features=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("\n[4/5] Evaluating...")
    y_pred  = clf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    print(f"\n  Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  CV Accuracy    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=3
    ))

    print("\n  Feature Importances:")
    importances = clf.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"    {feat:<40} {imp:.4f}  {bar}")

    print("\n[5/5] Saving artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENCODER_PATH)

    meta = {
        "model_type":       "RandomForestClassifier",
        "n_estimators":     120,
        "test_accuracy":    round(acc, 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std":  round(cv_scores.std(), 4),
        "features":         FEATURE_COLS,
        "classes":          list(le.classes_),
        "train_samples":    int(X_train.shape[0]),
        "test_samples":     int(X_test.shape[0]),
        "seed":             42,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model   -> {MODEL_PATH}")
    print(f"  Encoder -> {ENCODER_PATH}")
    print(f"  Meta    -> {META_PATH}")
    print("\n  Training complete.")
    print("=" * 60)
    return clf, le, meta

if __name__ == "__main__":
    train()
