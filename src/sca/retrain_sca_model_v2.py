#!/usr/bin/env python3
# retrain_sca_model_v2.py — Retrain with better diagnostics

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sca_dataset_balanced.npz",
                        help="Dataset file (npz format)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  SCA MODEL RETRAINING (v2)")
    print("=" * 60)
    
    # Load
    print(f"\n[1/5] Loading {args.data}...")
    data = np.load(args.data)
    X, y = data['X'], data['y']
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {np.bincount(y.astype(int))}")
    
    # Analyze class balance
    n_normal = np.sum(y == 0)
    n_leakage = np.sum(y == 1)
    print(f"  Balance: {n_normal} normal, {n_leakage} leakage")
    
    if abs(n_normal - n_leakage) > 50:
        print(f"  ⚠ WARNING: Imbalanced classes (ratio {n_normal/n_leakage:.2f})")
    
    # Split
    print("\n[2/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Grid search for best C value
    print("\n[3/5] Hyperparameter tuning (GridSearchCV)...")
    pipeline_base = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    
    param_grid = {"svm__C": [0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(pipeline_base, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"  Best C: {grid.best_params_['svm__C']}")
    print(f"  Best CV F1: {grid.best_score_:.3f}")
    
    pipeline = grid.best_estimator_
    
    # Evaluate
    print("\n[4/5] Evaluation...")
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy:  {test_acc:.2%}")
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"  ROC AUC:        {auc:.3f}")
    except:
        print(f"  ROC AUC:        (N/A)")
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["NORMAL", "LEAKAGE"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"    {'Pred NORMAL':>12s}  {'Pred LEAKAGE':>12s}")
    print(f"  True NORMAL   {cm[0,0]:>12d}  {cm[0,1]:>12d}")
    print(f"  True LEAKAGE  {cm[1,0]:>12d}  {cm[1,1]:>12d}")
    
    # Interpret results
    print("\n" + "=" * 60)
    print("  INTERPRETATION")
    print("=" * 60)
    
    if test_acc < 0.6:
        print(f"\n✗ POOR PERFORMANCE ({test_acc:.0%})")
        print("  Traces may not have sufficient signal difference")
        print("  Try: Increase CPU load intensity during collection")
    elif test_acc < 0.8:
        print(f"\n~ ACCEPTABLE ({test_acc:.0%})")
        print("  Model learned something but needs more data")
    else:
        print(f"\n✓ GOOD PERFORMANCE ({test_acc:.0%})")
        print("  Model is ready for deployment")
    
    # Save
    print("\n[5/5] Saving model...")
    import os
    os.makedirs("enrollment_data", exist_ok=True)
    joblib.dump(pipeline, "enrollment_data/sca_svm.joblib")
    
    print(f"✓ Model saved: enrollment_data/sca_svm.joblib")
    print(f"\nNext: python3 sca_module.py test")


if __name__ == "__main__":
    main()
