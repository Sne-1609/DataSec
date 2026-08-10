#!/usr/bin/env python3
# puf_audit.py — PUF Adversarial Audit Runner
# ============================================================
# Standalone script that:
#   1. Connects to the PUF (real hardware or simulator)
#   2. Runs the XGBoost modeling attack
#   3. Computes the Reliability Score
#   4. Triggers the AdaptiveController if vulnerable
#   5. Writes a JSON audit report
#   6. Exits with code 0 (SECURE/WARNING) or 2 (VULNERABLE)
#
# Designed to run automatically after device enrollment:
#   python3 enrollment.py enroll pi-alice --audit
#
# Or manually at any time:
#   python3 puf_audit.py                         # default 5000 CRPs, k=4
#   python3 puf_audit.py --crps 20000 --chains 4
#   python3 puf_audit.py --history               # show past audit results
#   python3 puf_audit.py --rotate                # force-rotate challenge mask
# ============================================================

import argparse
import json
import os
import sys
import time
from datetime import datetime

from puf_attacker import (
    run_audit, AdaptiveController, append_audit_log, load_audit_log,
    Verdict, ReliabilityScore,
    DEFAULT_XOR_CHAINS, DEFAULT_NOISE_RATE,
    AUDIT_LOG_PATH, CONTROLLER_STATE,
)


class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    END    = "\033[0m"


# ============================================================
# Report writer
# ============================================================

def write_report(score: ReliabilityScore,
                 controller: AdaptiveController,
                 path: str = "puf_audit_report.json"):
    """Write a detailed JSON report for this audit run."""
    report = {
        "audit_time":       datetime.utcnow().isoformat() + "Z",
        "reliability":      score.to_dict(),
        "adaptive_ctrl": {
            "rotation_count":       controller.rotation_count,
            "rotation_mask":        controller.rotation_mask.hex(),
            "vulnerability_events": len(controller.vulnerability_history),
        },
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  {C.CYAN}Audit report saved → {path}{C.END}")
    return report


# ============================================================
# History display
# ============================================================

def print_history(path: str = AUDIT_LOG_PATH):
    """Print a summary table of all past audit runs."""
    history = load_audit_log(path)

    if not history:
        print(f"  {C.YELLOW}No audit history found at {path}{C.END}")
        return

    print(f"\n{'='*70}")
    print(f"  PUF AUDIT HISTORY  ({len(history)} runs)")
    print(f"{'='*70}")
    print(f"  {'#':>3}  {'Date':19}  {'Chains':>6}  "
          f"{'CRPs':>7}  {'Accuracy':>8}  {'Score':>6}  {'Verdict'}")
    print(f"  {'-'*65}")

    for i, h in enumerate(history, 1):
        ts      = datetime.fromtimestamp(h["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        verdict = h["verdict"]
        colour  = {
            "SECURE":     C.GREEN,
            "WARNING":    C.YELLOW,
            "VULNERABLE": C.RED,
        }.get(verdict, "")
        print(f"  {i:>3}  {ts}  "
              f"{h['xor_chains']:>6}  "
              f"{h['n_train_crps'] + h['n_test_crps']:>7,}  "
              f"{h['test_accuracy']:>8.4f}  "
              f"{h['reliability_score']:>6.4f}  "
              f"{colour}{verdict}{C.END}")

    # Trend
    if len(history) >= 2:
        recent = history[-3:]
        scores = [h["reliability_score"] for h in recent]
        trend  = scores[-1] - scores[0]
        symbol = "↑" if trend > 0.01 else ("↓" if trend < -0.01 else "→")
        colour = C.GREEN if trend > 0.01 else (C.RED if trend < -0.01 else C.CYAN)
        print(f"\n  Trend (last {len(recent)} runs): "
              f"{colour}{symbol} {trend:+.4f}{C.END}")

    print(f"{'='*70}")


# ============================================================
# Force rotation
# ============================================================

def force_rotate(device_secret: bytes = b""):
    """Manually rotate the challenge-space mask."""
    ctrl = AdaptiveController(device_secret=device_secret)

    dummy_score = ReliabilityScore.compute(
        test_acc=0.0, train_acc=0.0,
        cv_mean=0.0, cv_std=0.0, roc=0.5,
        n_train=0, n_test=0,
        challenge_bits=64, xor_chains=0, noise_rate=0.0,
    )
    dummy_score.notes = "Manual rotation requested by operator"
    ctrl.rotate(dummy_score)
    print(f"  {C.GREEN}✓ Challenge mask rotated (count={ctrl.rotation_count}){C.END}")


# ============================================================
# Load device secret from enrollment data (if available)
# ============================================================

def load_device_secret() -> bytes:
    """
    Try to derive a device-specific secret from the PUF enrollment
    data. Falls back to an empty bytes object so the controller
    still works without enrollment data.
    """
    try:
        from puf_module import get_puf_response
        return get_puf_response(b"adaptive_ctrl_secret")[:16]
    except ImportError:
        pass

    # Try enrollment helper_data as fallback
    try:
        import json as _json
        with open("enrollment/helper_data.json", "r") as f:
            hd = _json.load(f)
        # Use a hash of any field in helper_data as device secret
        raw = json.dumps(hd, sort_keys=True).encode()
        import hashlib
        return hashlib.sha256(raw).digest()[:16]
    except Exception:
        pass

    return b""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PUF Adversarial Audit — XGBoost Reliability Scorer"
    )
    parser.add_argument("--crps",    type=int,   default=5000,
                        help="Challenge-Response Pairs to collect (default 5000)")
    parser.add_argument("--chains",  type=int,   default=DEFAULT_XOR_CHAINS,
                        help=f"XOR chains k (default {DEFAULT_XOR_CHAINS})")
    parser.add_argument("--noise",   type=float, default=DEFAULT_NOISE_RATE,
                        help=f"Simulated noise rate (default {DEFAULT_NOISE_RATE})")
    parser.add_argument("--cv",      type=int,   default=5,
                        help="Cross-validation folds (default 5)")
    parser.add_argument("--history", action="store_true",
                        help="Show audit history and exit")
    parser.add_argument("--rotate",  action="store_true",
                        help="Force-rotate challenge mask and exit")
    parser.add_argument("--report",  default="puf_audit_report.json",
                        help="Path to write JSON report (default puf_audit_report.json)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save XGBoost model to disk")
    args = parser.parse_args()

    # ── History mode ──────────────────────────────────────
    if args.history:
        print_history()
        return 0

    # ── Force-rotate mode ─────────────────────────────────
    if args.rotate:
        device_secret = load_device_secret()
        force_rotate(device_secret)
        return 0

    # ── Full audit ────────────────────────────────────────
    print(f"\n{C.BOLD}{'#'*60}{C.END}")
    print(f"{C.BOLD}#  PUF Adversarial Audit — XGBoost Modeling Attack{C.END}")
    print(f"{C.BOLD}{'#'*60}{C.END}")
    print(f"\n  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CRPs       : {args.crps:,}")
    print(f"  XOR chains : {args.chains}")
    print(f"  Noise rate : {args.noise:.1%}")
    print(f"  CV folds   : {args.cv}")

    device_secret = load_device_secret()
    controller    = AdaptiveController(device_secret=device_secret)

    t0    = time.monotonic()
    score = run_audit(
        n_crps     = args.crps,
        k_chains   = args.chains,
        noise_rate = args.noise,
        cv_folds   = args.cv,
        save_model = not args.no_save,
        controller = controller,
        verbose    = True,
    )
    elapsed = time.monotonic() - t0

    # ── Persist results ───────────────────────────────────
    append_audit_log(score)
    write_report(score, controller, path=args.report)

    print(f"\n  Total audit time: {elapsed:.1f} s")
    print(f"  Rotations to date: {controller.rotation_count}")

    # ── Exit code ─────────────────────────────────────────
    # 0 = SECURE or WARNING (system can continue)
    # 2 = VULNERABLE (caller should halt or re-enroll)
    if score.verdict == Verdict.VULNERABLE:
        print(f"\n  {C.RED}{C.BOLD}EXIT 2 — VULNERABLE PUF DETECTED{C.END}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
