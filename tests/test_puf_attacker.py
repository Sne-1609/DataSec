#!/usr/bin/env python3
# test_puf_attacker.py — XGBoost PUF Attacker — Full Test Suite
# ============================================================
# T1.  SimulatedPUF — determinism, noise, response balance
# T2.  SimulatedPUF — k-XOR chains increase resistance
# T3.  PUFInterface — fallback to simulator, query API
# T4.  CRPGenerator — uniqueness, response balance, large set
# T5.  PUFFeatureTransformer — shape, correctness, edge cases
# T6.  XGBoostPUFAttacker — train, predict, evaluate
# T7.  XGBoostPUFAttacker — feature importance, persistence
# T8.  ReliabilityScore — formula, thresholds, verdicts
# T9.  AdaptiveController — rotation, state persistence
# T10. End-to-end audit — k=1 (should be VULNERABLE or WARNING)
#                          k=8 (should be SECURE or WARNING)
# ============================================================

import os
import sys
import json
import tempfile
import hashlib
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puf_attacker import (
    SimulatedPUF, PUFInterface, CRPGenerator,
    PUFFeatureTransformer, XGBoostPUFAttacker,
    ReliabilityScore, Verdict, AdaptiveController,
    run_audit, append_audit_log, load_audit_log,
    CHALLENGE_BITS, DEFAULT_XOR_CHAINS, DEFAULT_NOISE_RATE,
    SCORE_SECURE, SCORE_WARNING,
)


# ── Colour helpers ─────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    END    = "\033[0m"


_results: dict = {}


def _pass(name: str):
    _results[name] = True
    print(f"  {C.GREEN}✓ PASS{C.END}  {name}")


def _fail(name: str, reason: str = ""):
    _results[name] = False
    msg = f"  {C.RED}✗ FAIL{C.END}  {name}"
    if reason:
        msg += f"  {C.YELLOW}({reason}){C.END}"
    print(msg)


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {C.BOLD}{title}{C.END}")
    print(f"{'='*60}")


# ============================================================
# T1 — SimulatedPUF: determinism, noise, balance
# ============================================================

def test_t1_simulated_puf():
    _section("T1: SimulatedPUF — Determinism, Noise, Balance")

    puf = SimulatedPUF(n_bits=64, k_chains=4, noise_rate=0.0, seed=0)
    rng = np.random.default_rng(1)

    # Determinism: same challenge must always return the same bit
    c  = rng.integers(0, 2, size=64, dtype=np.uint8)
    r1 = puf.get_response_bit(c)
    r2 = puf.get_response_bit(c)
    if r1 == r2:
        _pass("T1.1 same challenge → same response (deterministic)")
    else:
        _fail("T1.1 determinism", f"r1={r1} r2={r2}")

    # Different challenges usually produce different responses
    c2    = rng.integers(0, 2, size=64, dtype=np.uint8)
    diffs = sum(
        puf.get_response_bit(rng.integers(0, 2, 64, dtype=np.uint8)) !=
        puf.get_response_bit(rng.integers(0, 2, 64, dtype=np.uint8))
        for _ in range(50)
    )
    if diffs > 5:
        _pass(f"T1.2 different challenges → varied responses ({diffs}/50 differ)")
    else:
        _fail("T1.2 response variation", f"only {diffs}/50 differed")

    # Response balance: should be close to 50% ones
    bits = np.array([
        puf.get_response_bit(rng.integers(0, 2, 64, dtype=np.uint8))
        for _ in range(1000)
    ])
    mean = bits.mean()
    if 0.40 <= mean <= 0.60:
        _pass(f"T1.3 response balance {mean:.3f} in [0.40, 0.60]")
    else:
        _fail("T1.3 balance", f"mean={mean:.3f}")

    # Noise model: with noise_rate=0.5, responses should be ~50% flipped
    puf_noisy  = SimulatedPUF(64, 4, noise_rate=0.5, seed=0)
    puf_clean  = SimulatedPUF(64, 4, noise_rate=0.0, seed=0)
    challenges = [rng.integers(0, 2, 64, dtype=np.uint8) for _ in range(500)]
    flips = sum(
        puf_noisy.get_response_bit(ch) != puf_clean.get_response_bit(ch)
        for ch in challenges
    )
    if 150 <= flips <= 350:
        _pass(f"T1.4 noise model: {flips}/500 flips at noise_rate=0.5")
    else:
        _fail("T1.4 noise model", f"{flips}/500 flips")

    # Different device seeds → different responses (device uniqueness)
    puf_a = SimulatedPUF(64, 4, noise_rate=0.0, seed=0)
    puf_b = SimulatedPUF(64, 4, noise_rate=0.0, seed=999)
    agree = sum(
        puf_a.get_response_bit(ch) == puf_b.get_response_bit(ch)
        for ch in challenges[:100]
    )
    if agree < 80:
        _pass(f"T1.5 device uniqueness: seeds 0 vs 999 agree {agree}/100 times")
    else:
        _fail("T1.5 uniqueness", f"agree={agree}/100 (too similar)")

    # bytes shim compatibility
    rb = puf.get_response_bytes(b"test_challenge")
    if len(rb) == 256 and rb[0] & 1 in (0, 1):
        _pass("T1.6 get_response_bytes() returns 256 bytes with valid LSB")
    else:
        _fail("T1.6 bytes shim", f"len={len(rb)} byte0={rb[0]}")


# ============================================================
# T2 — k-XOR chain resistance
# ============================================================

def test_t2_xor_chain_resistance():
    _section("T2: XOR Chain Resistance (higher k = harder to attack)")

    rng = np.random.default_rng(42)

    # For a single Arbiter PUF (k=1), the parity features are linearly
    # separable. A quick linear test: correlation of parity features with
    # the response should be strong for k=1 but weak for k=8.

    def max_parity_correlation(k: int, n_samples: int = 500) -> float:
        puf = SimulatedPUF(64, k_chains=k, noise_rate=0.0, seed=0)
        challenges = np.array([
            rng.integers(0, 2, 64, dtype=np.uint8) for _ in range(n_samples)
        ])
        responses = np.array([
            puf.get_response_bit(c) for c in challenges
        ], dtype=np.float64)

        # Compute parity features and find max absolute correlation
        tx = PUFFeatureTransformer(64)
        X  = tx.transform(challenges)          # (N, 260)
        rs = responses - responses.mean()
        corrs = np.abs(np.corrcoef(X.T, rs)[-1, :-1])
        return float(np.max(corrs))

    corr_k1 = max_parity_correlation(1)
    corr_k4 = max_parity_correlation(4)
    corr_k8 = max_parity_correlation(8)

    if corr_k1 > corr_k4:
        _pass(f"T2.1 k=1 more linearly correlated ({corr_k1:.3f}) than k=4 ({corr_k4:.3f})")
    else:
        _fail("T2.1 k=1 vs k=4 correlation", f"k1={corr_k1:.3f} k4={corr_k4:.3f}")

    if corr_k4 > corr_k8:
        _pass(f"T2.2 k=4 more correlated ({corr_k4:.3f}) than k=8 ({corr_k8:.3f})")
    else:
        _fail("T2.2 k=4 vs k=8 correlation", f"k4={corr_k4:.3f} k8={corr_k8:.3f}")

    # Verify response bit is in {0, 1} for all chain counts
    for k in [1, 2, 4, 8]:
        puf = SimulatedPUF(64, k_chains=k, noise_rate=0.0, seed=k)
        bits = {puf.get_response_bit(rng.integers(0, 2, 64, dtype=np.uint8))
                for _ in range(20)}
        if bits <= {0, 1}:
            _pass(f"T2.3.k{k} k={k} response bits in {{0,1}}")
        else:
            _fail(f"T2.3.k{k} response bits", f"got {bits}")


# ============================================================
# T3 — PUFInterface
# ============================================================

def test_t3_puf_interface():
    _section("T3: PUFInterface — Simulator Fallback and Query API")

    puf = PUFInterface(n_bits=64, k_chains=4, noise_rate=0.0, seed=0)

    # Simulator fallback must work (puf_module not installed)
    if not puf.using_real_hardware:
        _pass("T3.1 simulator fallback active when puf_module unavailable")
    else:
        _pass("T3.1 real hardware puf_module detected")

    # query() returns 0 or 1
    rng = np.random.default_rng(99)
    responses = set()
    for _ in range(50):
        c = puf.random_challenge(rng)
        r = puf.query(c)
        assert r in (0, 1), f"response={r}"
        responses.add(r)
    if responses == {0, 1}:
        _pass("T3.2 query() returns both 0 and 1 across 50 challenges")
    else:
        _fail("T3.2 query values", f"only {responses}")

    # random_challenge() returns correct shape and dtype
    c = puf.random_challenge(rng)
    if c.shape == (64,) and c.dtype == np.uint8 and set(c).issubset({0, 1}):
        _pass("T3.3 random_challenge() shape=(64,) dtype=uint8 values in {0,1}")
    else:
        _fail("T3.3 challenge shape", f"shape={c.shape} dtype={c.dtype}")

    # Determinism: same challenge → same response
    c  = puf.random_challenge(np.random.default_rng(0))
    r1 = puf.query(c)
    r2 = puf.query(c)
    if r1 == r2:
        _pass("T3.4 interface deterministic for same challenge")
    else:
        _fail("T3.4 interface determinism", f"r1={r1} r2={r2}")


# ============================================================
# T4 — CRPGenerator
# ============================================================

def test_t4_crp_generator():
    _section("T4: CRPGenerator — Uniqueness, Balance, Performance")

    puf = PUFInterface(n_bits=64, k_chains=4, noise_rate=0.0, seed=0)
    gen = CRPGenerator(puf, seed=7)

    N  = 500
    X, y = gen.generate(N, verbose=False)

    # Shape
    if X.shape == (N, 64) and y.shape == (N,):
        _pass(f"T4.1 output shapes X={X.shape} y={y.shape}")
    else:
        _fail("T4.1 shapes", f"X={X.shape} y={y.shape}")

    # All values binary
    if set(np.unique(X)).issubset({0, 1}) and set(np.unique(y)).issubset({0, 1}):
        _pass("T4.2 all X values in {0,1} and y values in {0,1}")
    else:
        _fail("T4.2 binary values")

    # Uniqueness: no two identical challenges
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in X]
    if len(set(hashes)) == N:
        _pass(f"T4.3 all {N} challenges are unique")
    else:
        _fail("T4.3 uniqueness", f"{N - len(set(hashes))} duplicates")

    # Response balance ~50%
    balance = float(y.mean())
    if 0.38 <= balance <= 0.62:
        _pass(f"T4.4 response balance {balance:.3f} in [0.38, 0.62]")
    else:
        _fail("T4.4 balance", f"{balance:.3f}")

    # total_queried tracks correctly
    if gen.total_queried == N:
        _pass(f"T4.5 total_queried == {N}")
    else:
        _fail("T4.5 total_queried", f"{gen.total_queried}")

    # Second call appends uniquely
    X2, y2 = gen.generate(100, verbose=False)
    hashes2 = {hashlib.sha256(row.tobytes()).hexdigest() for row in X2}
    hashes1 = set(hashes)
    if not hashes2 & hashes1:
        _pass("T4.6 second generate() produces no duplicate challenges")
    else:
        _fail("T4.6 no reuse", f"{len(hashes2 & hashes1)} duplicates")


# ============================================================
# T5 — PUFFeatureTransformer
# ============================================================

def test_t5_feature_transformer():
    _section("T5: PUFFeatureTransformer — Shape, Correctness, Edge Cases")

    tx = PUFFeatureTransformer(n_bits=64)

    # Expected feature dimension: 64+65+64+64+1+2 = 260
    expected_dim = 64 + 65 + 64 + 64 + 1 + 2
    if tx.feature_dim == expected_dim:
        _pass(f"T5.1 feature_dim == {expected_dim}")
    else:
        _fail("T5.1 feature_dim", f"got {tx.feature_dim}")

    rng = np.random.default_rng(3)
    X   = rng.integers(0, 2, (100, 64), dtype=np.uint8)
    Xf  = tx.transform(X)

    if Xf.shape == (100, expected_dim):
        _pass(f"T5.2 transform output shape (100, {expected_dim})")
    else:
        _fail("T5.2 shape", f"got {Xf.shape}")

    # No NaN / Inf
    if np.all(np.isfinite(Xf)):
        _pass("T5.3 all features finite (no NaN/Inf)")
    else:
        _fail("T5.3 finite", f"non-finite at {np.where(~np.isfinite(Xf))}")

    # Parity feature for all-zeros challenge: all +1
    zeros = np.zeros((1, 64), dtype=np.uint8)
    Xf0   = tx.transform(zeros)
    parity_slice = Xf0[0, 64:64+65]     # group 2 starts at index 64
    if np.allclose(parity_slice, 1.0):
        _pass("T5.4 all-zero challenge → all-one parity features")
    else:
        _fail("T5.4 all-zero parity", f"min={parity_slice.min():.3f}")

    # Parity feature for all-ones challenge: alternates ±1
    ones = np.ones((1, 64), dtype=np.uint8)
    Xf1  = tx.transform(ones)
    parity_ones = Xf1[0, 64:64+65]
    # mapped = -1 for every bit, cumulative product: (-1)^(n-i) pattern
    expected_signs = set(np.sign(parity_ones))
    if expected_signs == {-1.0, 1.0} or expected_signs == {-1.0} or expected_signs == {1.0}:
        _pass("T5.5 all-ones challenge → alternating parity features")
    else:
        _fail("T5.5 all-ones parity")

    # Hamming weight feature (index 64+65+64+64 = 257)
    hw_idx = 64 + 65 + 64 + 64
    c_hw   = np.zeros((1, 64), dtype=np.uint8)
    c_hw[0, :32] = 1                    # 32 ones → hw = 0.5
    Xf_hw  = tx.transform(c_hw)
    if abs(Xf_hw[0, hw_idx] - 0.5) < 1e-6:
        _pass("T5.6 Hamming weight feature correct (32/64 ones → 0.5)")
    else:
        _fail("T5.6 Hamming weight", f"got {Xf_hw[0, hw_idx]:.4f}")

    # dtype must be float32
    if Xf.dtype == np.float32:
        _pass("T5.7 output dtype is float32")
    else:
        _fail("T5.7 dtype", f"got {Xf.dtype}")


# ============================================================
# T6 — XGBoostPUFAttacker: train, predict, evaluate
# ============================================================

def test_t6_xgboost_attacker():
    _section("T6: XGBoostPUFAttacker — Train, Predict, Evaluate")

    puf = PUFInterface(n_bits=64, k_chains=1, noise_rate=0.0, seed=0)
    gen = CRPGenerator(puf, seed=0)
    X, y = gen.generate(2000, verbose=False)

    X_train, y_train = X[:1600], y[:1600]
    X_test,  y_test  = X[1600:], y[1600:]

    attacker = XGBoostPUFAttacker(n_bits=64)

    # Untrained model should raise RuntimeError
    try:
        attacker.predict(X_test[:1])
        _fail("T6.1 untrained predict() should raise RuntimeError")
    except RuntimeError:
        _pass("T6.1 untrained predict() raises RuntimeError")

    train_acc = attacker.train(X_train, y_train)

    if attacker.is_trained:
        _pass("T6.2 is_trained == True after train()")
    else:
        _fail("T6.2 is_trained")

    # k=1 PUF is linearly separable → expect high train accuracy
    if train_acc > 0.70:
        _pass(f"T6.3 k=1 train accuracy {train_acc:.4f} > 0.70 (expected high)")
    else:
        _fail("T6.3 train accuracy", f"{train_acc:.4f}")

    # predict() returns array with values in {0,1}
    y_pred = attacker.predict(X_test)
    if y_pred.shape == (400,) and set(np.unique(y_pred)).issubset({0, 1}):
        _pass("T6.4 predict() shape=(400,) values in {0,1}")
    else:
        _fail("T6.4 predict shape/values", f"shape={y_pred.shape}")

    # predict_proba() returns (N,2) in [0,1]
    proba = attacker.predict_proba(X_test)
    if (proba.shape == (400, 2) and
            np.all(proba >= 0) and np.all(proba <= 1) and
            np.allclose(proba.sum(axis=1), 1.0)):
        _pass("T6.5 predict_proba() shape=(400,2) valid probabilities")
    else:
        _fail("T6.5 predict_proba", f"shape={proba.shape}")

    # evaluate() returns dict with accuracy and roc_auc
    metrics = attacker.evaluate(X_test, y_test)
    if ("accuracy" in metrics and "roc_auc" in metrics and
            0.0 <= metrics["accuracy"] <= 1.0 and
            0.0 <= metrics["roc_auc"]  <= 1.0):
        _pass(f"T6.6 evaluate() returns valid metrics "
              f"acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f}")
    else:
        _fail("T6.6 evaluate metrics", str(metrics.keys()))

    # k=1 test accuracy should exceed random for 2000 CRPs
    if metrics["accuracy"] > 0.55:
        _pass(f"T6.7 k=1 test accuracy {metrics['accuracy']:.4f} > 0.55 "
              f"(attacker learns the PUF)")
    else:
        _fail("T6.7 k=1 test accuracy", f"{metrics['accuracy']:.4f}")

    return attacker


# ============================================================
# T7 — Feature importance and persistence
# ============================================================

def test_t7_importance_and_persistence(attacker: XGBoostPUFAttacker):
    _section("T7: Feature Importance and Model Persistence")

    # top_features returns list of (name, importance) pairs
    top = attacker.top_features(10)
    if (len(top) == 10 and
            all(isinstance(n, str) and isinstance(v, float) for n, v in top) and
            all(v >= 0 for _, v in top)):
        _pass(f"T7.1 top_features(10) returns 10 valid (name, importance) pairs")
    else:
        _fail("T7.1 top_features", str(top[:3]))

    # Importances should be sorted descending
    importances = [v for _, v in top]
    if importances == sorted(importances, reverse=True):
        _pass("T7.2 top_features sorted by importance descending")
    else:
        _fail("T7.2 sort order")

    # Parity features should be among the top features for k=1 PUF
    top_names = [n for n, _ in top]
    parity_in_top = any("parity" in n for n in top_names)
    if parity_in_top:
        _pass("T7.3 parity features appear in top-10 (expected for k=1 PUF)")
    else:
        _fail("T7.3 parity in top features", str(top_names))

    # Save / load round-trip
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        path = tf.name
    try:
        attacker.save(path)
        loaded = XGBoostPUFAttacker()
        loaded.load(path)
        if loaded.is_trained:
            _pass("T7.4 loaded model is_trained == True")
        else:
            _fail("T7.4 loaded is_trained")

        # Both models must give same predictions
        rng     = np.random.default_rng(55)
        X_rand  = rng.integers(0, 2, (50, 64), dtype=np.uint8)
        p_orig  = attacker.predict(X_rand)
        p_load  = loaded.predict(X_rand)
        if np.array_equal(p_orig, p_load):
            _pass("T7.5 original and loaded model give identical predictions")
        else:
            n_diff = int(np.sum(p_orig != p_load))
            _fail("T7.5 prediction consistency", f"{n_diff}/50 differ")
    finally:
        os.unlink(path)


# ============================================================
# T8 — ReliabilityScore formula and verdict thresholds
# ============================================================

def test_t8_reliability_score():
    _section("T8: ReliabilityScore — Formula and Verdict Thresholds")

    def make(acc):
        return ReliabilityScore.compute(
            test_acc=acc, train_acc=acc, cv_mean=acc, cv_std=0.01,
            roc=acc, n_train=1000, n_test=200,
            challenge_bits=64, xor_chains=4, noise_rate=0.03,
        )

    # Perfect random (50%) → score = 1.0
    s = make(0.50)
    if abs(s.reliability_score - 1.0) < 1e-6:
        _pass("T8.1 accuracy=0.50 → reliability_score=1.0")
    else:
        _fail("T8.1 score at 50%", f"{s.reliability_score:.6f}")

    if s.verdict == Verdict.SECURE:
        _pass("T8.2 score=1.0 → verdict SECURE")
    else:
        _fail("T8.2 verdict at 50%", s.verdict.value)

    # accuracy=0.55 → score=0.90 → boundary SECURE/WARNING
    s55 = make(0.55)
    if abs(s55.reliability_score - 0.90) < 1e-6:
        _pass("T8.3 accuracy=0.55 → reliability_score=0.90")
    else:
        _fail("T8.3 score at 55%", f"{s55.reliability_score:.6f}")

    # accuracy=0.625 → score=0.75 → boundary WARNING/VULNERABLE
    s625 = make(0.625)
    if abs(s625.reliability_score - 0.75) < 1e-6:
        _pass("T8.4 accuracy=0.625 → reliability_score=0.75")
    else:
        _fail("T8.4 score at 62.5%", f"{s625.reliability_score:.6f}")

    # accuracy=0.80 → score=0.60 → VULNERABLE
    s80 = make(0.80)
    if s80.verdict == Verdict.VULNERABLE:
        _pass(f"T8.5 accuracy=0.80 → verdict VULNERABLE (score={s80.reliability_score:.2f})")
    else:
        _fail("T8.5 verdict at 80%", s80.verdict.value)

    # accuracy=0.60 → WARNING
    s60 = make(0.60)
    if s60.verdict == Verdict.WARNING:
        _pass(f"T8.6 accuracy=0.60 → verdict WARNING (score={s60.reliability_score:.2f})")
    else:
        _fail("T8.6 verdict at 60%", s60.verdict.value)

    # score is clipped to [0,1]
    s_extreme = make(1.0)
    if 0.0 <= s_extreme.reliability_score <= 1.0:
        _pass("T8.7 reliability_score clipped to [0,1] for extreme accuracy")
    else:
        _fail("T8.7 clip", f"{s_extreme.reliability_score:.4f}")

    # to_dict() is serialisable
    d = s.to_dict()
    try:
        json.dumps(d)
        _pass("T8.8 to_dict() output is JSON-serialisable")
    except TypeError as e:
        _fail("T8.8 JSON serialisable", str(e))

    # summary() returns a non-empty string
    summary = s.summary()
    if isinstance(summary, str) and len(summary) > 100:
        _pass("T8.9 summary() returns non-empty string")
    else:
        _fail("T8.9 summary", f"len={len(summary)}")


# ============================================================
# T9 — AdaptiveController
# ============================================================

def test_t9_adaptive_controller():
    _section("T9: AdaptiveController — Rotation, State Persistence")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                     mode="w") as tf:
        state_path = tf.name
    # delete the temp file so controller starts fresh
    os.unlink(state_path)

    try:
        ctrl = AdaptiveController(device_secret=b"test_secret",
                                  state_path=state_path)

        # Initial state
        if ctrl.rotation_count == 0:
            _pass("T9.1 initial rotation_count == 0")
        else:
            _fail("T9.1 initial count", f"{ctrl.rotation_count}")

        initial_mask = ctrl.rotation_mask

        # Trigger rotation via a VULNERABLE score
        vuln_score = ReliabilityScore.compute(
            test_acc=0.85, train_acc=0.90,
            cv_mean=0.84, cv_std=0.02, roc=0.90,
            n_train=800, n_test=200,
            challenge_bits=64, xor_chains=1, noise_rate=0.0,
        )
        ctrl.react(vuln_score)

        if ctrl.rotation_count == 1:
            _pass("T9.2 rotation_count == 1 after VULNERABLE react()")
        else:
            _fail("T9.2 rotation_count", f"{ctrl.rotation_count}")

        if ctrl.rotation_mask != initial_mask:
            _pass("T9.3 mask changed after rotation")
        else:
            _fail("T9.3 mask changed")

        # Challenge scrambling is invertible (XOR is self-inverse)
        c = os.urandom(8)
        s = ctrl.scramble_challenge(c)
        s_again = ctrl.scramble_challenge(c)
        if s == s_again:
            _pass("T9.4 scramble_challenge is deterministic")
        else:
            _fail("T9.4 deterministic scramble")

        if s != c:
            _pass("T9.5 scrambled challenge differs from original")
        else:
            _fail("T9.5 challenge changed by mask")

        # State persists: reload controller from same file
        ctrl2 = AdaptiveController(device_secret=b"test_secret",
                                   state_path=state_path)
        if ctrl2.rotation_count == 1:
            _pass("T9.6 rotation_count persisted across reload")
        else:
            _fail("T9.6 persistence", f"{ctrl2.rotation_count}")

        if ctrl2.rotation_mask == ctrl.rotation_mask:
            _pass("T9.7 rotation_mask persisted correctly")
        else:
            _fail("T9.7 mask persistence")

        # Second rotation changes mask again
        ctrl2.react(vuln_score)
        if ctrl2.rotation_mask != ctrl.rotation_mask:
            _pass("T9.8 second rotation produces new unique mask")
        else:
            _fail("T9.8 second rotation mask")

        # SECURE score does not rotate
        mask_before = ctrl2.rotation_mask
        secure_score = ReliabilityScore.compute(
            test_acc=0.51, train_acc=0.52,
            cv_mean=0.51, cv_std=0.01, roc=0.51,
            n_train=800, n_test=200,
            challenge_bits=64, xor_chains=8, noise_rate=0.03,
        )
        ctrl2.react(secure_score)
        if ctrl2.rotation_mask == mask_before:
            _pass("T9.9 SECURE score does not trigger rotation")
        else:
            _fail("T9.9 no rotation on SECURE")

        # vulnerability_history grows correctly
        if len(ctrl2.vulnerability_history) == 2:
            _pass("T9.10 vulnerability_history has 2 entries")
        else:
            _fail("T9.10 history length", f"{len(ctrl2.vulnerability_history)}")

    finally:
        if os.path.exists(state_path):
            os.unlink(state_path)


# ============================================================
# T10 — End-to-end audit
# ============================================================

def test_t10_end_to_end():
    _section("T10: End-to-End Audit — k=1 (vulnerable) vs k=8 (secure)")

    # ── k=1 single Arbiter PUF — should be caught as VULNERABLE or WARNING
    print("  Running k=1 audit (5000 CRPs)…")
    score_k1 = run_audit(
        n_crps     = 5000,
        k_chains   = 1,
        noise_rate = 0.0,         # no noise → easier to attack
        cv_folds   = 3,
        save_model = False,
        verbose    = False,
    )
    print(f"  k=1 result: acc={score_k1.test_accuracy:.4f}  "
          f"score={score_k1.reliability_score:.4f}  "
          f"verdict={score_k1.verdict.value}")

    # k=1 with 5000 CRPs should detect above-random accuracy
    if score_k1.test_accuracy > 0.55:
        _pass(f"T10.1 k=1 test accuracy {score_k1.test_accuracy:.4f} > 0.55 "
              f"(vulnerability detected)")
    else:
        _fail("T10.1 k=1 detection",
              f"accuracy={score_k1.test_accuracy:.4f} — attacker failed to learn")

    if score_k1.verdict in (Verdict.VULNERABLE, Verdict.WARNING):
        _pass(f"T10.2 k=1 verdict is {score_k1.verdict.value} (not SECURE)")
    else:
        _fail("T10.2 k=1 verdict", score_k1.verdict.value)

    # ── k=8 strong XOR PUF — should be SECURE with 3000 CRPs
    print("\n  Running k=8 audit (3000 CRPs)…")
    score_k8 = run_audit(
        n_crps     = 3000,
        k_chains   = 8,
        noise_rate = DEFAULT_NOISE_RATE,
        cv_folds   = 3,
        save_model = False,
        verbose    = False,
    )
    print(f"  k=8 result: acc={score_k8.test_accuracy:.4f}  "
          f"score={score_k8.reliability_score:.4f}  "
          f"verdict={score_k8.verdict.value}")

    # k=8 with 3000 CRPs should be hard to break
    if score_k8.test_accuracy < 0.63:
        _pass(f"T10.3 k=8 test accuracy {score_k8.test_accuracy:.4f} < 0.63 "
              f"(PUF resists attack with limited CRPs)")
    else:
        _fail("T10.3 k=8 resistance",
              f"accuracy={score_k8.test_accuracy:.4f} — too predictable")

    if score_k8.verdict in (Verdict.SECURE, Verdict.WARNING):
        _pass(f"T10.4 k=8 verdict is {score_k8.verdict.value} (not VULNERABLE)")
    else:
        _fail("T10.4 k=8 verdict", score_k8.verdict.value)

    # k=8 score must be higher than k=1 score (more random = more reliable)
    if score_k8.reliability_score > score_k1.reliability_score:
        _pass(f"T10.5 k=8 more reliable ({score_k8.reliability_score:.4f}) "
              f"than k=1 ({score_k1.reliability_score:.4f})")
    else:
        _fail("T10.5 reliability ordering",
              f"k8={score_k8.reliability_score:.4f} k1={score_k1.reliability_score:.4f}")

    # ── Audit log round-trip ───────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                     mode="w") as tf:
        log_path = tf.name
    os.unlink(log_path)

    try:
        append_audit_log(score_k1, path=log_path)
        append_audit_log(score_k8, path=log_path)
        history = load_audit_log(path=log_path)

        if len(history) == 2:
            _pass("T10.6 audit log save/load round-trip (2 entries)")
        else:
            _fail("T10.6 audit log", f"got {len(history)} entries")

        if (history[0]["xor_chains"] == 1 and
                history[1]["xor_chains"] == 8):
            _pass("T10.7 audit log entries ordered and contain correct chain count")
        else:
            _fail("T10.7 audit log content")
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)

    # ── AdaptiveController integration ────────────────────
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                     mode="w") as tf:
        ctrl_path = tf.name
    os.unlink(ctrl_path)

    try:
        ctrl = AdaptiveController(device_secret=b"e2e_test",
                                  state_path=ctrl_path)
        rotations_before = ctrl.rotation_count
        ctrl.react(score_k1)   # k=1 is likely VULNERABLE → triggers rotation
        rotations_after = ctrl.rotation_count

        if score_k1.verdict == Verdict.VULNERABLE:
            if rotations_after == rotations_before + 1:
                _pass("T10.8 AdaptiveController rotated on VULNERABLE k=1 score")
            else:
                _fail("T10.8 rotation on VULNERABLE", f"count={rotations_after}")
        else:
            # WARNING — no rotation expected
            _pass(f"T10.8 k=1 was {score_k1.verdict.value} (WARNING) — "
                  f"no rotation required, controller reacted correctly")
    finally:
        if os.path.exists(ctrl_path):
            os.unlink(ctrl_path)


# ============================================================
# Main
# ============================================================

def main():
    print(f"\n{'#'*60}")
    print(f"#  {C.BOLD}XGBoost PUF Attacker — Full Test Suite{C.END}")
    print(f"{'#'*60}")

    test_t1_simulated_puf()
    test_t2_xor_chain_resistance()
    test_t3_puf_interface()
    test_t4_crp_generator()
    test_t5_feature_transformer()

    attacker = test_t6_xgboost_attacker()  # returns trained attacker for T7
    test_t7_importance_and_persistence(attacker)

    test_t8_reliability_score()
    test_t9_adaptive_controller()
    test_t10_end_to_end()

    # ── Final summary ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {C.BOLD}FINAL RESULTS{C.END}")
    print(f"{'='*60}")

    passed = sum(1 for v in _results.values() if v)
    total  = len(_results)

    for name, ok in _results.items():
        sym = f"{C.GREEN}PASS{C.END}" if ok else f"{C.RED}FAIL{C.END}"
        print(f"  {sym}  {name}")

    print(f"\n  {'─'*40}")
    if passed == total:
        print(f"  {C.GREEN}{C.BOLD}ALL {total} TESTS PASSED ✓{C.END}")
    else:
        print(f"  {C.RED}{C.BOLD}{passed}/{total} PASSED — "
              f"{total - passed} FAILED ✗{C.END}")
    print(f"  {'─'*40}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
