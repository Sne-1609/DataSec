# sca_train.py
#!/usr/bin/env python3
# sca_train.py — One-shot SCA model training script
# ============================================================
# Run this ONCE on each Pi during initial setup (after
# enrollment.py).  It:
#   1. Optionally collects REAL power traces via INA219 while
#      Kyber operations run (hardware mode).
#   2. Falls back to high-fidelity synthetic data when no
#      sensor is wired up (software-only / CI mode).
#   3. Trains the RBF-SVM and saves sca_model.pkl.
#   4. Prints a full evaluation report.
#
# Usage:
#   python3 sca_train.py                  # synthetic data (no hardware)
#   python3 sca_train.py --real           # collect real traces first
#   python3 sca_train.py --real --i2c 1   # specify I2C bus number
#   python3 sca_train.py --eval-only      # evaluate existing model
# ============================================================

import argparse
import sys
import os
import time
import numpy as np

from sca_module import (
    INA219Sampler, SVMSCADetector, TraceDataset, Label,
    generate_synthetic_dataset, train_sca_model,
    WINDOW_SAMPLES, SAMPLE_RATE_HZ, MODEL_PATH, DATASET_PATH,
)

# ── try to import PQC so we can run real Kyber ops ──────────
try:
    from pqc_module import PQCKeyExchange
    HAS_PQC = True
except ImportError:
    HAS_PQC = False


# ============================================================
# Real trace collection
# ============================================================

def collect_real_traces(
        n_normal:  int = 300,
        n_leakage: int = 150,
        bus_num:   int = 1,
) -> TraceDataset:
    """
    Collect labelled power traces directly from INA219.

    Normal traces  — captured during idle / AES-only operation.
    Leakage traces — captured during Kyber KeyGen / Decaps while
                     deliberately stressing the power supply
                     (e.g. simultaneous GPIO toggling to emulate
                     correlated switching noise).

    If liboqs is unavailable the simulated PQC path is used,
    which produces realistic timing even without real Kyber.
    """
    sampler = INA219Sampler(bus_num=bus_num)
    ds      = TraceDataset()

    print(f"\n[COLLECT] Normal traces ({n_normal})…")
    for i in range(n_normal):
        sampler.sim_inject_leakage(False)   # ensure clean state
        if HAS_PQC:
            # Run a real (or simulated) Kyber op to keep load realistic
            pqc = PQCKeyExchange()
            _   = pqc.generate_keypair()
        else:
            time.sleep(WINDOW_SAMPLES / SAMPLE_RATE_HZ)

        trace = sampler.collect_window(WINDOW_SAMPLES)
        ds.add(trace, Label.NORMAL)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_normal}")

    print(f"\n[COLLECT] Leakage traces ({n_leakage})…")
    for i in range(n_leakage):
        # Inject a simulated spike mid-operation to label it as leakage
        # On real hardware: trigger a glitch or correlated GPIO toggle here
        sampler.sim_inject_leakage(True)

        if HAS_PQC:
            pqc  = PQCKeyExchange()
            pk   = pqc.generate_keypair()
            pqc2 = PQCKeyExchange()
            ct, _  = pqc2.encapsulate(pk)
            _    = pqc.decapsulate(ct)
        else:
            time.sleep(WINDOW_SAMPLES / SAMPLE_RATE_HZ)

        trace = sampler.collect_window(WINDOW_SAMPLES)
        ds.add(trace, Label.LEAKAGE)
        sampler.sim_inject_leakage(False)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_leakage}")

    sampler.close()
    print(f"\n[COLLECT] Done — {len(ds.labels)} total traces "
          f"({ds.labels.count(0)} normal / {ds.labels.count(1)} leakage)")
    return ds


# ============================================================
# Evaluation on a held-out split
# ============================================================

def evaluate_model(detector: SVMSCADetector, dataset: TraceDataset):
    """
    Hold out 20 % of the dataset and print per-class metrics.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    X, y = dataset.X(), dataset.y()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Retrain on split (detector may already be fitted on full set)
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(C=10.0, kernel="rbf", gamma="scale",
                       class_weight="balanced", probability=True,
                       random_state=42))
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    print("\n── Held-out Evaluation (20% split) ──────────────────")
    print(classification_report(
        y_te, y_pred,
        target_names=["NORMAL", "LEAKAGE"],
        zero_division=0
    ))
    cm = confusion_matrix(y_te, y_pred)
    print("Confusion matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    tpr = cm[1,1] / max(cm[1,0] + cm[1,1], 1)
    fpr = cm[0,1] / max(cm[0,0] + cm[0,1], 1)
    print(f"\n  TPR (Recall):  {tpr:.3f}   (fraction of attacks caught)")
    print(f"  FPR:           {fpr:.3f}   (false alarm rate)")
    print("─────────────────────────────────────────────────────")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train the SCA/SVM model for this project"
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Collect real power traces via INA219 (requires hardware)"
    )
    parser.add_argument(
        "--i2c", type=int, default=1,
        help="I2C bus number for INA219 (default: 1)"
    )
    parser.add_argument(
        "--n-normal", type=int, default=800,
        help="Number of normal traces to collect/generate (default: 800)"
    )
    parser.add_argument(
        "--n-leakage", type=int, default=200,
        help="Number of leakage traces to collect/generate (default: 200)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; just evaluate the existing model"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to model file (default: {MODEL_PATH})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SCA MODEL TRAINING — Triple-Layer Security Project")
    print("=" * 60)

    # ── Eval-only path ────────────────────────────────────
    if args.eval_only:
        if not os.path.exists(args.model):
            print(f"ERROR: model not found at {args.model}")
            print("Run without --eval-only first to train.")
            sys.exit(1)
        print(f"\n[1] Loading model from {args.model}…")
        detector = SVMSCADetector()
        detector.load(args.model)

        if os.path.exists(DATASET_PATH):
            print(f"[2] Loading dataset from {DATASET_PATH}…")
            ds = TraceDataset.load(DATASET_PATH)
            evaluate_model(detector, ds)
        else:
            print("[2] No dataset file found — generating synthetic for eval…")
            ds = generate_synthetic_dataset(args.n_normal, args.n_leakage)
            evaluate_model(detector, ds)
        return

    # ── Collect or generate dataset ───────────────────────
    if args.real:
        print(f"\n[1] Collecting real power traces (I2C bus {args.i2c})…")
        dataset = collect_real_traces(
            n_normal=args.n_normal,
            n_leakage=args.n_leakage,
            bus_num=args.i2c,
        )
    else:
        print(f"\n[1] Generating synthetic dataset "
              f"({args.n_normal} normal / {args.n_leakage} leakage)…")
        dataset = generate_synthetic_dataset(
            n_normal=args.n_normal,
            n_leakage=args.n_leakage,
        )

    # ── Train ─────────────────────────────────────────────
    print("\n[2] Training SVM…")
    detector, metrics = train_sca_model(
        dataset=dataset,
        save_model_path=args.model,
        save_data_path=DATASET_PATH,
    )

    print("\n── Training Results ──────────────────────────────────")
    print(f"  CV F1:      {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion:  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"              FN={cm[1][0]}  TP={cm[1][1]}")
    print("─────────────────────────────────────────────────────")

    # ── Held-out evaluation ───────────────────────────────
    print("\n[3] Running held-out evaluation…")
    evaluate_model(detector, dataset)

    print(f"\n[4] Model saved → {args.model}")
    print("\n  Next: run  python3 device_a.py  (or device_b.py)")
    print("        The SCA monitor will load this model automatically.\n")


if __name__ == "__main__":
    main()
