#sca-collect
#!/usr/bin/env python3
# sca_collect.py — Real-hardware SCA trace collector
# ============================================================
# Run this ONCE on your Pi 5 to build a labelled dataset from
# actual INA219 power measurements during Kyber operations.
#
# It records two kinds of traces:
#   NORMAL  — clean Kyber keygen / encapsulate / decapsulate
#   LEAKAGE — same operations with artificially injected GPIO
#             toggling (simulates a fault-injection or leaky
#             implementation) so the SVM learns real Pi 5 noise.
#
# Usage:
#   python3 sca_collect.py --samples 500 --out sca_dataset.npz
#   python3 sca_collect.py --samples 500 --out sca_dataset.npz --retrain
# ============================================================

import argparse
import time
import os
import sys

import numpy as np

from sca_modulev2 import (
    INA219Sampler, TraceDataset, Label,
    WINDOW_SAMPLES, SAMPLE_RATE_HZ,
    train_sca_model, MODEL_PATH, DATASET_PATH,
)

# ── ANSI colours ───────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    END    = "\033[0m"


def collect_normal_traces(
        sampler:    INA219Sampler,
        dataset:    TraceDataset,
        n:          int,
        use_kyber:  bool = True,
) -> None:
    """
    Collect n NORMAL traces.
    If use_kyber is True and liboqs is available, runs real Kyber
    key-gen for each window so the power trace is authentic.
    Falls back to a 200 ms sleep (still draws real Pi 5 idle power).
    """
    print(f"\n{C.CYAN}Collecting {n} NORMAL traces…{C.END}")

    try:
        from pqc_module import PQCKeyExchange
        pqc_available = True
    except ImportError:
        pqc_available = False

    for i in range(n):
        if use_kyber and pqc_available:
            # Run a real Kyber keygen concurrently with the window capture
            # (keygen takes ~1–3 ms on Pi 5; window is ~200 ms so we
            #  interleave multiple ops inside a single window)
            def _kyber_ops():
                kex = PQCKeyExchange()
                pk  = kex.generate_keypair()
                kex2 = PQCKeyExchange()
                ct, _ = kex2.encapsulate(pk)
                kex.decapsulate(ct)

            import threading
            t = threading.Thread(target=_kyber_ops, daemon=True)
            t.start()
            trace = sampler.collect_window(WINDOW_SAMPLES, SAMPLE_RATE_HZ)
            t.join(timeout=1.0)
        else:
            time.sleep(0.01)
            trace = sampler.collect_window(WINDOW_SAMPLES, SAMPLE_RATE_HZ)

        dataset.add(trace, Label.NORMAL)

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  {C.GREEN}✓ {i+1}/{n} normal traces collected{C.END}")


def collect_leakage_traces(
        sampler:   INA219Sampler,
        dataset:   TraceDataset,
        n:         int,
) -> None:
    """
    Collect n LEAKAGE traces.

    Leakage is induced by toggling a GPIO pin rapidly during the
    crypto operation (models a real glitch / EM emanation).
    Falls back to the simulator's spike injection when no GPIO.
    """
    print(f"\n{C.YELLOW}Collecting {n} LEAKAGE traces…{C.END}")

    gpio_available = False
    gpio_pin       = 17   # BCM numbering — change to any free GPIO

    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(gpio_pin, GPIO.OUT, initial=GPIO.LOW)
        gpio_available = True
        print(f"  GPIO {gpio_pin} available — using real GPIO toggling")
    except Exception:
        print(f"  GPIO not available — using simulator spike injection")
        sampler.sim_inject_leakage(True)

    def _toggle_gpio():
        """Toggle GPIO at ~5 kHz for one window duration."""
        if not gpio_available:
            return
        import RPi.GPIO as GPIO
        duration = WINDOW_SAMPLES / SAMPLE_RATE_HZ
        end_time = time.monotonic() + duration
        while time.monotonic() < end_time:
            GPIO.output(gpio_pin, GPIO.HIGH)
            time.sleep(0.0001)
            GPIO.output(gpio_pin, GPIO.LOW)
            time.sleep(0.0001)

    for i in range(n):
        import threading
        t = threading.Thread(target=_toggle_gpio, daemon=True)
        t.start()
        trace = sampler.collect_window(WINDOW_SAMPLES, SAMPLE_RATE_HZ)
        t.join(timeout=1.0)
        dataset.add(trace, Label.LEAKAGE)

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  {C.RED}⚡ {i+1}/{n} leakage traces collected{C.END}")

    if gpio_available:
        import RPi.GPIO as GPIO
        GPIO.cleanup()
    else:
        sampler.sim_inject_leakage(False)


def main():
    parser = argparse.ArgumentParser(
        description="Collect labelled SCA traces from INA219 on Pi 5"
    )
    parser.add_argument("--samples",  type=int, default=500,
                        help="Number of traces PER class (default 500)")
    parser.add_argument("--ratio",    type=float, default=0.25,
                        help="Leakage fraction (default 0.25 = 25%%)")
    parser.add_argument("--out",      default=DATASET_PATH,
                        help=f"Output dataset path (default {DATASET_PATH})")
    parser.add_argument("--retrain",  action="store_true",
                        help="Retrain SVM after collecting")
    parser.add_argument("--no-kyber", action="store_true",
                        help="Skip Kyber ops; use idle power traces only")
    args = parser.parse_args()

    n_total   = args.samples
    n_leakage = max(1, int(n_total * args.ratio))
    n_normal  = n_total - n_leakage

    print(f"\n{C.BOLD}=== SCA Trace Collector ==={C.END}")
    print(f"  Total samples : {n_total}")
    print(f"  Normal        : {n_normal}")
    print(f"  Leakage       : {n_leakage}")
    print(f"  Output        : {args.out}")

    sampler = INA219Sampler()
    dataset = TraceDataset()

    # If an existing dataset exists, load it and append
    if os.path.exists(args.out):
        print(f"\n{C.YELLOW}Existing dataset found — appending to it.{C.END}")
        dataset = TraceDataset.load(args.out)
        print(f"  Loaded {len(dataset.labels)} existing samples.")

    try:
        collect_normal_traces(sampler, dataset, n_normal,
                               use_kyber=not args.no_kyber)
        collect_leakage_traces(sampler, dataset, n_leakage)
    finally:
        sampler.close()

    dataset.save(args.out)

    label_counts = {
        "NORMAL":  dataset.labels.count(0),
        "LEAKAGE": dataset.labels.count(1),
    }
    print(f"\n{C.GREEN}Dataset saved → {args.out}{C.END}")
    print(f"  Total samples: {len(dataset.labels)}")
    print(f"  NORMAL:        {label_counts['NORMAL']}")
    print(f"  LEAKAGE:       {label_counts['LEAKAGE']}")

    if args.retrain:
        print(f"\n{C.CYAN}Retraining SVM on new dataset…{C.END}")
        detector, metrics = train_sca_model(
            dataset=dataset,
            save_model_path=MODEL_PATH,
            save_data_path=args.out,
        )
        print(f"\n{C.GREEN}✓ Retrain complete{C.END}")
        print(f"  CV F1:     {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
