#!/usr/bin/env python3
# puf_attacker.py — XGBoost PUF Modeling Attack & Adversarial Auditor
# ============================================================
# Implements a full "modeling attack" on the Physical Unclonable
# Function (PUF) as a binary classification task:
#
#   Input  : n-bit challenge vector
#   Output : 1-bit predicted response
#
# The attack pipeline:
#   1. CRPGenerator  — collect (challenge, response_bit) pairs
#                      from the real PUF or the SimulatedPUF
#   2. PUFFeatureTransformer
#                    — convert raw challenge bits into a 260-dim
#                      feature vector (parity, cumulative XOR,
#                      Hamming weight, half-parities)
#   3. XGBoostPUFAttacker
#                    — train / cross-validate / predict
#   4. ReliabilityScorer
#                    — translate attacker accuracy into a PUF
#                      Reliability Score (1.0 = coin-flip secure,
#                      0.0 = fully compromised)
#   5. AdaptiveController
#                    — if score < VULNERABLE_THRESHOLD, rotate
#                      the challenge-space mask and alert
#
# PUF model (SimulatedPUF):
#   Implements a k-XOR Arbiter PUF using the standard linear
#   threshold model from PUF literature (Rührmair et al.).
#   k=4 chains → realistic attack difficulty on 5,000 CRPs.
#   k=8 chains → near-random accuracy even with 50,000 CRPs.
#
# Reliability Score:
#   score = 1 - 2 * |accuracy - 0.5|
#   score = 1.0  →  accuracy = 50%  →  perfectly random  →  SECURE
#   score = 0.0  →  accuracy = 100% →  fully predictable →  COMPROMISED
#
# CLI:
#   python3 puf_attacker.py attack  --crps 5000 --chains 4
#   python3 puf_attacker.py audit   --crps 10000
#   python3 puf_attacker.py demo
# ============================================================

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import logging
import pickle
import argparse
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Tuple, List, Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix,
)
import xgboost as xgb

# ── Logging ────────────────────────────────────────────────
log = logging.getLogger("puf_attacker")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "[PUF %(levelname)s %(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_h)


# ============================================================
# Constants
# ============================================================

CHALLENGE_BITS      = 64        # standard Arbiter PUF challenge length
DEFAULT_XOR_CHAINS  = 4         # k-XOR PUF chains (higher = more secure)
DEFAULT_NOISE_RATE  = 0.03      # 3% measurement noise (realistic PUF)

# Reliability score thresholds
SCORE_SECURE        = 0.90      # accuracy ≤ 55%  → SECURE ✓
SCORE_WARNING       = 0.75      # accuracy ≤ 62.5% → WARNING ⚠
# Below SCORE_WARNING              accuracy > 62.5% → VULNERABLE ✗

# File paths
MODEL_PATH          = "puf_xgb_model.pkl"
AUDIT_LOG_PATH      = "puf_audit_log.json"
CONTROLLER_STATE    = "puf_controller_state.json"


# ============================================================
# Security verdict
# ============================================================

class Verdict(str, Enum):
    SECURE     = "SECURE"        # score >= SCORE_SECURE
    WARNING    = "WARNING"       # SCORE_WARNING <= score < SCORE_SECURE
    VULNERABLE = "VULNERABLE"    # score < SCORE_WARNING


@dataclass
class ReliabilityScore:
    """
    Result object produced after one full attack run.

    reliability_score:
        1.0 = coin-flip random (perfectly secure)
        0.0 = perfectly predictable (fully compromised)
        Formula: 1 - 2 * |accuracy - 0.5|

    test_accuracy:
        Raw XGBoost prediction accuracy on held-out CRPs.
        ~0.50 = good, >> 0.55 = bad.
    """
    test_accuracy:      float
    train_accuracy:     float
    cv_accuracy_mean:   float
    cv_accuracy_std:    float
    roc_auc:            float
    reliability_score:  float
    verdict:            Verdict
    n_train_crps:       int
    n_test_crps:        int
    challenge_bits:     int
    xor_chains:         int
    noise_rate:         float
    timestamp:          float    = field(default_factory=time.time)
    notes:              str      = ""

    @classmethod
    def compute(cls, test_acc: float, train_acc: float,
                cv_mean: float, cv_std: float, roc: float,
                n_train: int, n_test: int,
                challenge_bits: int, xor_chains: int,
                noise_rate: float) -> "ReliabilityScore":
        score = 1.0 - 2.0 * abs(test_acc - 0.5)
        score = float(np.clip(score, 0.0, 1.0))

        if score >= SCORE_SECURE:
            verdict = Verdict.SECURE
        elif score >= SCORE_WARNING:
            verdict = Verdict.WARNING
        else:
            verdict = Verdict.VULNERABLE

        return cls(
            test_accuracy=test_acc,
            train_accuracy=train_acc,
            cv_accuracy_mean=cv_mean,
            cv_accuracy_std=cv_std,
            roc_auc=roc,
            reliability_score=score,
            verdict=verdict,
            n_train_crps=n_train,
            n_test_crps=n_test,
            challenge_bits=challenge_bits,
            xor_chains=xor_chains,
            noise_rate=noise_rate,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def summary(self) -> str:
        C_RED    = "\033[91m"
        C_YELLOW = "\033[93m"
        C_GREEN  = "\033[92m"
        C_BOLD   = "\033[1m"
        C_END    = "\033[0m"

        colour = {
            Verdict.SECURE:     C_GREEN,
            Verdict.WARNING:    C_YELLOW,
            Verdict.VULNERABLE: C_RED,
        }[self.verdict]

        icon = {
            Verdict.SECURE:     "✓",
            Verdict.WARNING:    "⚠",
            Verdict.VULNERABLE: "✗",
        }[self.verdict]

        lines = [
            f"\n{'='*60}",
            f"  PUF RELIABILITY AUDIT REPORT",
            f"{'='*60}",
            f"  Challenge bits    : {self.challenge_bits}",
            f"  XOR chains (k)    : {self.xor_chains}",
            f"  Noise rate        : {self.noise_rate:.1%}",
            f"  Training CRPs     : {self.n_train_crps:,}",
            f"  Test CRPs         : {self.n_test_crps:,}",
            f"  Train accuracy    : {self.train_accuracy:.4f}  ({self.train_accuracy:.1%})",
            f"  CV accuracy       : {self.cv_accuracy_mean:.4f} ± {self.cv_accuracy_std:.4f}",
            f"  Test accuracy     : {self.test_accuracy:.4f}  ({self.test_accuracy:.1%})",
            f"  ROC-AUC           : {self.roc_auc:.4f}",
            f"  Reliability score : {self.reliability_score:.4f}",
            f"",
            f"  {colour}{C_BOLD}{icon}  VERDICT: {self.verdict.value}{C_END}",
        ]

        if self.verdict == Verdict.SECURE:
            lines.append(
                f"  {C_GREEN}  Attacker accuracy ≈ random coin flip. PUF is secure.{C_END}"
            )
        elif self.verdict == Verdict.WARNING:
            lines.append(
                f"  {C_YELLOW}  Marginal vulnerability detected. Consider increasing"
                f" XOR chains or rotating the challenge space.{C_END}"
            )
        else:
            lines.append(
                f"  {C_RED}  Mathematical vulnerability detected! Adaptive controller"
                f" will rotate the challenge space.{C_END}"
            )

        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ============================================================
# Simulated PUF  (k-XOR Arbiter PUF)
# ============================================================

class SimulatedPUF:
    """
    Mathematical model of a k-XOR Arbiter PUF.

    Based on the linear threshold model (Rührmair et al., 2010).
    Each Arbiter chain has n+1 Gaussian delay weights. The response
    is determined by the sign of the dot product of the parity
    feature vector with the weight vector, XOR'd across k chains.

    Reference:
      U. Rührmair et al., "Modeling attacks on physical unclonable
      functions", ACM CCS 2010.
    """

    def __init__(self,
                 n_bits:     int   = CHALLENGE_BITS,
                 k_chains:   int   = DEFAULT_XOR_CHAINS,
                 noise_rate: float = DEFAULT_NOISE_RATE,
                 seed:       int   = 0):
        """
        n_bits    : challenge length in bits
        k_chains  : number of XOR chains (higher = harder to attack)
        noise_rate: probability that a response bit is flipped (noise)
        seed      : random seed for reproducible device simulation
        """
        self.n_bits     = n_bits
        self.k_chains   = k_chains
        self.noise_rate = noise_rate

        # Device-specific weight matrices — Gaussian delay differences
        # Shape: (k_chains, n_bits + 1)  — +1 for the final arbiter
        rng = np.random.default_rng(seed)
        self._weights = rng.standard_normal((k_chains, n_bits + 1)).astype(np.float64)
        self._noise_rng = np.random.default_rng(seed + 1)

        log.debug("SimulatedPUF: n=%d k=%d noise=%.2f%%",
                  n_bits, k_chains, noise_rate * 100)

    # ── Feature transform ─────────────────────────────────

    @staticmethod
    def parity_features(challenge_bits: np.ndarray) -> np.ndarray:
        """
        Convert a (n,) binary challenge into the (n+1,) parity
        feature vector Φ used by the Arbiter PUF linear model.

        Φ_i = ∏_{j=i}^{n-1} (1 - 2*c_j)   for i = 0…n-1
        Φ_n = 1.0  (bias)

        Maps {0,1} → {+1,-1} then takes cumulative products
        from right to left.
        """
        n = len(challenge_bits)
        mapped = (1 - 2 * challenge_bits).astype(np.float64)    # {0,1}→{+1,-1}
        phi = np.empty(n + 1, dtype=np.float64)
        phi[n] = 1.0
        for i in range(n - 1, -1, -1):
            phi[i] = phi[i + 1] * mapped[i]
        return phi

    # ── Single-challenge response ─────────────────────────

    def get_response_bit(self, challenge_bits: np.ndarray) -> int:
        """
        Evaluate the k-XOR PUF on one (n,) binary challenge.
        Returns 0 or 1 (with optional noise).
        """
        phi = self.parity_features(challenge_bits)

        # Each chain produces a 1-bit response
        response = 0
        for chain_idx in range(self.k_chains):
            delay_diff = np.dot(self._weights[chain_idx], phi)
            chain_bit  = 1 if delay_diff > 0 else 0
            response  ^= chain_bit          # XOR across all chains

        # Measurement noise
        if self.noise_rate > 0 and self._noise_rng.random() < self.noise_rate:
            response ^= 1

        return response

    def get_response_bytes(self, challenge: bytes) -> bytes:
        """
        Compatibility shim: accepts a bytes challenge (like the real
        puf_module.get_puf_response), returns 256 bytes where bit 0
        of byte 0 is the Arbiter PUF response bit.
        """
        # Expand challenge to exactly n_bits bits
        challenge_int = int.from_bytes(
            hashlib.sha256(challenge).digest()[:self.n_bits // 8],
            "big"
        )
        challenge_bits = np.array(
            [(challenge_int >> (self.n_bits - 1 - i)) & 1
             for i in range(self.n_bits)],
            dtype=np.uint8,
        )
        response_bit = self.get_response_bit(challenge_bits)

        # Build 256-byte response (bit 0 of byte 0 carries the PUF bit)
        response_bytes = bytearray(
            hashlib.sha256(challenge + b"puf_expand").digest() * 8
        )
        response_bytes[0] = (response_bytes[0] & 0xFE) | response_bit
        return bytes(response_bytes)


# ============================================================
# PUF Interface  (real hardware OR simulator)
# ============================================================

class PUFInterface:
    """
    Unified interface for querying the PUF.

    Tries to import the real `puf_module.get_puf_response`.
    Falls back to SimulatedPUF when the module is unavailable
    (e.g., on a development machine without PUF hardware).

    Response bit extraction from a full PUF response (bytes):
        response_bit = puf_response_bytes[0] & 1   (LSB of byte 0)

    This is the standard convention: the first bit carries the
    arbiter decision from the first challenge application.
    """

    def __init__(self,
                 n_bits:    int   = CHALLENGE_BITS,
                 k_chains:  int   = DEFAULT_XOR_CHAINS,
                 noise_rate: float = DEFAULT_NOISE_RATE,
                 seed:      int   = 0):
        self.n_bits = n_bits
        self._real  = False
        self._sim   = SimulatedPUF(n_bits, k_chains, noise_rate, seed)

        try:
            from puf_module import get_puf_response as _real_get
            self._real_get = _real_get
            self._real     = True
            log.info("PUFInterface: using real puf_module.get_puf_response")
        except ImportError:
            log.info("PUFInterface: puf_module unavailable — using SimulatedPUF "
                     "(k=%d, noise=%.1f%%)", k_chains, noise_rate * 100)

    @property
    def using_real_hardware(self) -> bool:
        return self._real

    def query(self, challenge_bits: np.ndarray) -> int:
        """
        Query the PUF with a (n_bits,) binary challenge array.
        Returns the 1-bit response (0 or 1).
        """
        if self._real:
            # Pack challenge bits into bytes for puf_module API
            challenge_bytes = np.packbits(challenge_bits).tobytes()
            response_bytes  = self._real_get(challenge_bytes)
            return int(response_bytes[0]) & 1
        else:
            return self._sim.get_response_bit(challenge_bits)

    def random_challenge(self, rng: np.random.Generator) -> np.ndarray:
        """Generate one random n_bits challenge."""
        return rng.integers(0, 2, size=self.n_bits, dtype=np.uint8)


# ============================================================
# CRP Generator
# ============================================================

class CRPGenerator:
    """
    Collects Challenge-Response Pairs (CRPs) from the PUF.

    Each CRP is:
        challenge : (n_bits,) uint8 array  — uniformly random
        response  : int  0 or 1            — PUF hardware output

    The generator avoids challenge reuse by hashing each challenge
    and checking a set of used hashes. This matches the real-world
    adversary model where each CRP is obtained in a separate query.
    """

    def __init__(self,
                 puf:     PUFInterface,
                 seed:    int = 42):
        self._puf  = puf
        self._rng  = np.random.default_rng(seed)
        self._used: set = set()              # SHA-256 hashes of used challenges

    def _challenge_hash(self, c: np.ndarray) -> str:
        return hashlib.sha256(c.tobytes()).hexdigest()

    def generate(self,
                 n_crps:    int,
                 verbose:   bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate n_crps unique CRPs.

        Returns:
            challenges : (n_crps, n_bits) uint8 array
            responses  : (n_crps,)        int8  array  (values 0 or 1)
        """
        n  = self._puf.n_bits
        Xs = np.empty((n_crps, n),    dtype=np.uint8)
        ys = np.empty((n_crps,),      dtype=np.int8)

        collected = 0
        retries   = 0
        t0        = time.monotonic()

        while collected < n_crps:
            c = self._puf.random_challenge(self._rng)
            h = self._challenge_hash(c)

            if h in self._used:
                retries += 1
                if retries > n_crps * 10:
                    raise RuntimeError(
                        "CRPGenerator: too many retries — challenge space exhausted?"
                    )
                continue

            self._used.add(h)
            Xs[collected] = c
            ys[collected] = self._puf.query(c)
            collected += 1

            if verbose and collected % max(1, n_crps // 10) == 0:
                elapsed = time.monotonic() - t0
                rate    = collected / elapsed
                log.info("  CRPs collected: %d / %d  (%.0f/s)", collected, n_crps, rate)

        elapsed = time.monotonic() - t0
        log.info("Generated %d CRPs in %.2f s  (%.0f CRP/s)",
                 n_crps, elapsed, n_crps / elapsed)
        log.info("Response balance: %d ones / %d zeros  (%.1f%% ones)",
                 int(ys.sum()), int((1 - ys).sum()), 100 * ys.mean())

        return Xs, ys.astype(np.int32)

    @property
    def total_queried(self) -> int:
        return len(self._used)


# ============================================================
# PUF Feature Transformer
# ============================================================

class PUFFeatureTransformer:
    """
    Converts raw (n,) binary challenge vectors into a rich feature
    matrix for XGBoost.

    Feature groups (for n=64 challenge bits → 260 total features):

    Group 1 — Raw bits (64):
        The raw {0,1} challenge bits. Captures direct bit effects.

    Group 2 — Parity features (65):
        The standard linearisation of Arbiter PUF delay differences.
        Φ_i = ∏_{j=i}^{n-1} (1-2c_j), Φ_n = 1 (bias).
        A single Arbiter PUF is linearly separable in this space.
        XGBoost uses this to attack each chain independently.

    Group 3 — Cumulative XOR from left (64):
        Running XOR: x_i = c_0 ⊕ c_1 ⊕ … ⊕ c_i
        Captures the parity of challenge prefixes.

    Group 4 — Cumulative XOR from right (64):
        Running XOR: x_i = c_i ⊕ c_{i+1} ⊕ … ⊕ c_{n-1}
        Captures the parity of challenge suffixes.

    Group 5 — Normalised Hamming weight (1):
        hw = Σc_i / n.  Encodes global bit density.

    Group 6 — Half-parities (2):
        Parity of first half, parity of second half.
        Cheap proxies for inter-segment correlation.

    Total: 64 + 65 + 64 + 64 + 1 + 2 = 260 features.
    """

    def __init__(self, n_bits: int = CHALLENGE_BITS):
        self.n_bits       = n_bits
        self.feature_dim  = n_bits + (n_bits + 1) + n_bits + n_bits + 1 + 2

    def transform(self, challenges: np.ndarray) -> np.ndarray:
        """
        challenges : (N, n_bits) uint8 array
        Returns    : (N, feature_dim) float32 array
        """
        challenges = np.asarray(challenges, dtype=np.float32)
        N, n = challenges.shape
        assert n == self.n_bits, f"Expected {self.n_bits} bits, got {n}"

        # ── Group 1: Raw bits ──────────────────────────────
        raw = challenges                                        # (N, n)

        # ── Group 2: Parity features ───────────────────────
        mapped = 1.0 - 2.0 * challenges                        # {0,1}→{+1,-1}
        parity = np.ones((N, n + 1), dtype=np.float32)
        # Cumulative product from right: parity[:,i] = ∏_{j=i}^{n-1} mapped[:,j]
        for i in range(n - 1, -1, -1):
            parity[:, i] = parity[:, i + 1] * mapped[:, i]    # (N,)

        # ── Group 3: Cumulative XOR from left ──────────────
        cum_l = np.cumsum(challenges.astype(np.int32), axis=1) % 2  # (N, n)
        cum_l = cum_l.astype(np.float32)

        # ── Group 4: Cumulative XOR from right ─────────────
        cum_r = (np.cumsum(challenges[:, ::-1].astype(np.int32),
                           axis=1) % 2)[:, ::-1]               # (N, n)
        cum_r = cum_r.astype(np.float32)

        # ── Group 5: Normalised Hamming weight ─────────────
        hw = (np.sum(challenges, axis=1, keepdims=True) / n)   # (N, 1)

        # ── Group 6: Half-parities ─────────────────────────
        half  = n // 2
        p_lo  = (np.sum(challenges[:, :half],  axis=1,
                        keepdims=True).astype(np.int32) % 2).astype(np.float32)
        p_hi  = (np.sum(challenges[:, half:],  axis=1,
                        keepdims=True).astype(np.int32) % 2).astype(np.float32)

        X = np.hstack([raw, parity, cum_l, cum_r, hw, p_lo, p_hi])
        assert X.shape == (N, self.feature_dim), \
            f"Feature dim mismatch: {X.shape[1]} != {self.feature_dim}"
        return X


# ============================================================
# XGBoost PUF Attacker
# ============================================================

class XGBoostPUFAttacker:
    """
    XGBoost binary classifier trained to predict PUF responses.

    The XGBoost ensemble is better suited than logistic regression
    for XOR PUFs because:
    - It learns non-linear interactions between chains implicitly
    - It handles the XOR composition of multiple linear functions
    - Gradient boosting iteratively corrects residuals, fitting
      each misclassified CRP more tightly per round

    Hyperparameters are tuned for:
    - Challenge length: 64 bits
    - XOR chains: k = 4
    - CRP budget: 1,000 – 50,000
    - Hardware: Raspberry Pi 5 Cortex-A76 (4 cores)
    """

    # Hyperparameters
    _PARAMS = dict(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.80,
        colsample_bytree = 0.60,
        min_child_weight = 5,
        gamma            = 0.10,
        reg_alpha        = 0.10,
        reg_lambda       = 1.00,
        scale_pos_weight = 1,       # balanced classes expected
        tree_method      = "hist",  # histogram method — fast on Pi 5
        eval_metric      = "logloss",
        random_state     = 42,
        n_jobs           = -1,      # use all Pi 5 cores
        verbosity        = 0,
    )

    def __init__(self, n_bits: int = CHALLENGE_BITS):
        self.n_bits      = n_bits
        self._transformer = PUFFeatureTransformer(n_bits)
        self._model: Optional[xgb.XGBClassifier] = None
        self._trained    = False

    # ── Training ──────────────────────────────────────────

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val:   Optional[np.ndarray] = None,
              y_val:   Optional[np.ndarray] = None) -> float:
        """
        Train the XGBoost model on (challenge, response) pairs.

        X_train : (N, n_bits)  raw challenge bits
        y_train : (N,)         response bits {0, 1}
        Returns : train accuracy
        """
        Xf_train = self._transformer.transform(X_train)

        eval_set = [(Xf_train, y_train)]
        if X_val is not None and y_val is not None:
            Xf_val = self._transformer.transform(X_val)
            eval_set.append((Xf_val, y_val))

        self._model = xgb.XGBClassifier(**self._PARAMS)

        log.info("Training XGBoost on %d CRPs (%d features)…",
                 len(y_train), Xf_train.shape[1])
        t0 = time.monotonic()
        self._model.fit(
            Xf_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        elapsed = time.monotonic() - t0
        log.info("Training complete in %.2f s", elapsed)

        y_pred        = self._model.predict(Xf_train)
        train_accuracy = float(accuracy_score(y_train, y_pred))
        self._trained = True
        return train_accuracy

    # ── Cross-validation ──────────────────────────────────

    def cross_validate(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       n_folds: int = 5) -> Tuple[float, float]:
        """
        Stratified k-fold CV to estimate generalisation accuracy.
        Returns (mean_accuracy, std_accuracy).
        """
        Xf  = self._transformer.transform(X)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        mdl = xgb.XGBClassifier(**self._PARAMS)

        log.info("Running %d-fold cross-validation…", n_folds)
        scores = cross_val_score(mdl, Xf, y, cv=skf,
                                 scoring="accuracy", n_jobs=-1)
        log.info("CV accuracy: %.4f ± %.4f", scores.mean(), scores.std())
        return float(scores.mean()), float(scores.std())

    # ── Inference ─────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict response bits for challenge array (N, n_bits)."""
        self._require_trained()
        return self._model.predict(self._transformer.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (N, 2) for challenges."""
        self._require_trained()
        return self._model.predict_proba(self._transformer.transform(X))

    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> Dict[str, float]:
        """
        Full evaluation on a held-out test set.
        Returns dict with accuracy, roc_auc, and per-class metrics.
        """
        self._require_trained()
        Xf      = self._transformer.transform(X_test)
        y_pred  = self._model.predict(Xf)
        y_proba = self._model.predict_proba(Xf)[:, 1]

        acc  = float(accuracy_score(y_test, y_pred))
        auc  = float(roc_auc_score(y_test, y_proba))
        cm   = confusion_matrix(y_test, y_pred).tolist()
        rep  = classification_report(y_test, y_pred,
                                     target_names=["resp=0", "resp=1"],
                                     output_dict=True, zero_division=0)

        log.info("Test accuracy : %.4f  (%.1f%%)", acc, acc * 100)
        log.info("ROC-AUC       : %.4f", auc)
        log.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

        return {
            "accuracy": acc,
            "roc_auc":  auc,
            "confusion_matrix": cm,
            "report": rep,
        }

    # ── Feature importance ────────────────────────────────

    def top_features(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Return top_n most important features by XGBoost gain.
        Feature names follow the PUFFeatureTransformer layout.
        """
        self._require_trained()
        n = self.n_bits
        names: List[str] = (
            [f"raw_{i}"     for i in range(n)]        +   # 64
            [f"parity_{i}"  for i in range(n + 1)]    +   # 65
            [f"cxor_l_{i}"  for i in range(n)]        +   # 64
            [f"cxor_r_{i}"  for i in range(n)]        +   # 64
            ["hamming_wt"]                             +   # 1
            ["half_parity_lo", "half_parity_hi"]           # 2
        )
        importances = self._model.feature_importances_
        pairs = sorted(zip(names, importances),
                       key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    # ── Persistence ───────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        self._require_trained()
        with open(path, "wb") as f:
            pickle.dump((self._model, self.n_bits), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Model saved → %s", path)

    def load(self, path: str = MODEL_PATH):
        with open(path, "rb") as f:
            self._model, self.n_bits = pickle.load(f)
        self._transformer = PUFFeatureTransformer(self.n_bits)
        self._trained = True
        log.info("Model loaded ← %s", path)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _require_trained(self):
        if not self._trained:
            raise RuntimeError(
                "XGBoostPUFAttacker: model not trained. Call train() first."
            )


# ============================================================
# Adaptive Controller
# ============================================================

class AdaptiveController:
    """
    Reacts to PUF vulnerability assessments by rotating the
    challenge-space mask so any attacker model trained on the
    old challenge space becomes immediately invalid.

    Challenge scrambling:
        effective_challenge = raw_challenge ⊕ rotation_mask

    The rotation mask is derived from a device secret (the PUF's
    own enrollment response) and a rotation counter, so it cannot
    be predicted by an external adversary.

    State is persisted to CONTROLLER_STATE so it survives restarts.
    """

    def __init__(self,
                 device_secret: bytes = b"",
                 state_path:    str   = CONTROLLER_STATE):
        self._device_secret = device_secret
        self._state_path    = state_path
        self._state         = self._load_state()

    # ── State management ──────────────────────────────────

    def _default_state(self) -> dict:
        initial_mask = hashlib.sha256(
            self._device_secret + b"init_mask"
        ).digest()[:8]
        return {
            "rotation_count":    0,
            "rotation_mask":     initial_mask.hex(),
            "vulnerability_log": [],
            "last_score":        None,
        }

    def _load_state(self) -> dict:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return self._default_state()

    def _save_state(self):
        with open(self._state_path, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Challenge scrambling ──────────────────────────────

    @property
    def rotation_mask(self) -> bytes:
        return bytes.fromhex(self._state["rotation_mask"])

    def scramble_challenge(self, challenge: bytes) -> bytes:
        """XOR challenge with current rotation mask."""
        mask = self.rotation_mask
        # Extend mask to challenge length by repeating + hashing
        while len(mask) < len(challenge):
            mask += hashlib.sha256(mask).digest()[:8]
        mask = mask[:len(challenge)]
        return bytes(a ^ b for a, b in zip(challenge, mask))

    # ── Rotation ──────────────────────────────────────────

    def rotate(self, score: ReliabilityScore):
        """
        Rotate the challenge mask in response to a vulnerability.
        Logs the event and saves updated state.
        """
        old_mask = self._state["rotation_mask"]
        count    = self._state["rotation_count"] + 1

        # New mask = SHA-256(old_mask || device_secret || counter)
        new_mask = hashlib.sha256(
            bytes.fromhex(old_mask)
            + self._device_secret
            + count.to_bytes(4, "big")
        ).digest()[:8]

        self._state["rotation_mask"]  = new_mask.hex()
        self._state["rotation_count"] = count
        self._state["last_score"]     = score.to_dict()
        self._state["vulnerability_log"].append({
            "timestamp":    time.time(),
            "rotation":     count,
            "old_mask":     old_mask,
            "new_mask":     new_mask.hex(),
            "verdict":      score.verdict.value,
            "test_acc":     score.test_accuracy,
            "rel_score":    score.reliability_score,
        })

        self._save_state()

        C_RED  = "\033[91m"
        C_BOLD = "\033[1m"
        C_END  = "\033[0m"
        print(f"\n{C_RED}{C_BOLD}[ADAPTIVE CONTROLLER]{C_END}")
        print(f"  Rotation #{count}: challenge-space mask rotated.")
        print(f"  Old mask: {old_mask}")
        print(f"  New mask: {new_mask.hex()}")
        print(f"  Any attacker model trained on the previous CRPs")
        print(f"  is now invalid — all future challenges are remapped.")

    # ── React to audit result ─────────────────────────────

    def react(self, score: ReliabilityScore):
        """
        Evaluate the ReliabilityScore and trigger rotation if needed.
        """
        if score.verdict == Verdict.VULNERABLE:
            log.warning("VULNERABLE PUF detected (score=%.3f) — rotating.",
                        score.reliability_score)
            self.rotate(score)
        elif score.verdict == Verdict.WARNING:
            log.warning("WARNING: marginal PUF reliability (score=%.3f). "
                        "Consider increasing XOR chains.", score.reliability_score)
        else:
            log.info("PUF SECURE (score=%.3f). No rotation needed.",
                     score.reliability_score)

    @property
    def rotation_count(self) -> int:
        return self._state["rotation_count"]

    @property
    def vulnerability_history(self) -> List[dict]:
        return list(self._state["vulnerability_log"])


# ============================================================
# Full audit pipeline
# ============================================================

def run_audit(
        n_crps:      int   = 5000,
        test_frac:   float = 0.20,
        k_chains:    int   = DEFAULT_XOR_CHAINS,
        noise_rate:  float = DEFAULT_NOISE_RATE,
        cv_folds:    int   = 5,
        save_model:  bool  = True,
        model_path:  str   = MODEL_PATH,
        controller:  Optional[AdaptiveController] = None,
        seed:        int   = 42,
        verbose:     bool  = True,
) -> ReliabilityScore:
    """
    End-to-end PUF modeling attack and reliability audit.

    Steps:
      1. Connect to PUF (real or simulated)
      2. Collect n_crps challenge-response pairs
      3. Split into train / test
      4. Train XGBoost attacker on train set
      5. Cross-validate for generalisation estimate
      6. Evaluate on held-out test set
      7. Compute ReliabilityScore and Verdict
      8. Trigger AdaptiveController if score indicates vulnerability
      9. Save model and return ReliabilityScore

    Returns: ReliabilityScore dataclass
    """
    # ── 1. PUF interface ───────────────────────────────────
    puf = PUFInterface(
        n_bits=CHALLENGE_BITS,
        k_chains=k_chains,
        noise_rate=noise_rate,
        seed=seed,
    )
    if verbose:
        hw_str = "REAL HARDWARE" if puf.using_real_hardware else "SIMULATOR"
        print(f"\n  PUF source : {hw_str}")
        print(f"  XOR chains : {k_chains}")
        print(f"  Noise rate : {noise_rate:.1%}")
        print(f"  Total CRPs : {n_crps:,}")

    # ── 2. Collect CRPs ────────────────────────────────────
    gen = CRPGenerator(puf, seed=seed)
    X, y = gen.generate(n_crps, verbose=verbose)

    # ── 3. Train/test split ────────────────────────────────
    n_test  = max(100, int(n_crps * test_frac))
    n_train = n_crps - n_test

    # Shuffle before splitting
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(n_crps)
    X, y   = X[idx], y[idx]

    X_train, y_train = X[:n_train], y[:n_train]
    X_test,  y_test  = X[n_train:], y[n_train:]

    if verbose:
        print(f"\n  Train CRPs : {n_train:,}")
        print(f"  Test CRPs  : {n_test:,}")
        print(f"  Response balance (train): {y_train.mean():.3f}")

    # ── 4. Train XGBoost ───────────────────────────────────
    attacker    = XGBoostPUFAttacker(CHALLENGE_BITS)
    train_acc   = attacker.train(X_train, y_train, X_test, y_test)

    # ── 5. Cross-validate ──────────────────────────────────
    cv_mean, cv_std = attacker.cross_validate(X_train, y_train, n_folds=cv_folds)

    # ── 6. Evaluate on test set ────────────────────────────
    eval_metrics = attacker.evaluate(X_test, y_test)
    test_acc     = eval_metrics["accuracy"]
    roc_auc      = eval_metrics["roc_auc"]

    # ── 7. Compute reliability score ───────────────────────
    score = ReliabilityScore.compute(
        test_acc    = test_acc,
        train_acc   = train_acc,
        cv_mean     = cv_mean,
        cv_std      = cv_std,
        roc         = roc_auc,
        n_train     = n_train,
        n_test      = n_test,
        challenge_bits = CHALLENGE_BITS,
        xor_chains  = k_chains,
        noise_rate  = noise_rate,
    )

    if verbose:
        print(score.summary())

        # Feature importance
        top = attacker.top_features(10)
        print("\n  Top 10 most important XGBoost features:")
        for feat_name, importance in top:
            bar = "█" * int(importance * 400)
            print(f"    {feat_name:20s}  {importance:.5f}  {bar}")

    # ── 8. Adaptive controller ─────────────────────────────
    if controller is not None:
        controller.react(score)

    # ── 9. Save model ──────────────────────────────────────
    if save_model:
        attacker.save(model_path)

    return score


# ============================================================
# Audit log
# ============================================================

def append_audit_log(score: ReliabilityScore,
                     path:  str = AUDIT_LOG_PATH):
    """Append one ReliabilityScore entry to the JSON audit log."""
    history = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                history = json.load(f)
        except Exception:
            pass
    history.append(score.to_dict())
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Audit log updated → %s  (%d entries)", path, len(history))


def load_audit_log(path: str = AUDIT_LOG_PATH) -> List[dict]:
    """Load and return the full audit history."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# CLI
# ============================================================

def _cmd_attack(args):
    """Run the modeling attack and report."""
    print(f"\n{'#'*60}")
    print(f"#  XGBoost PUF Modeling Attack")
    print(f"#  CRPs={args.crps}  chains={args.chains}  noise={args.noise:.1%}")
    print(f"{'#'*60}")

    ctrl  = AdaptiveController()
    score = run_audit(
        n_crps     = args.crps,
        k_chains   = args.chains,
        noise_rate = args.noise,
        cv_folds   = args.cv,
        controller = ctrl,
    )
    append_audit_log(score)
    return 0 if score.verdict != Verdict.VULNERABLE else 1


def _cmd_audit(args):
    """Run repeated audits and print history."""
    print(f"\n{'#'*60}")
    print(f"#  PUF Reliability Audit  (full report)")
    print(f"{'#'*60}")

    ctrl  = AdaptiveController()
    score = run_audit(
        n_crps     = args.crps,
        k_chains   = args.chains,
        noise_rate = args.noise,
        cv_folds   = args.cv,
        controller = ctrl,
    )
    append_audit_log(score)

    # Print history
    history = load_audit_log()
    if len(history) > 1:
        print(f"\n  Audit history ({len(history)} runs):")
        for i, h in enumerate(history[-5:], 1):
            print(f"    Run {i}: verdict={h['verdict']}"
                  f"  acc={h['test_accuracy']:.3f}"
                  f"  score={h['reliability_score']:.3f}")

    return 0


def _cmd_demo(args):
    """Demo: compare secure (k=8) vs vulnerable (k=1) PUF."""
    print(f"\n{'#'*60}")
    print(f"#  PUF Modeling Attack Demo")
    print(f"{'#'*60}")

    configs = [
        (1, "Single Arbiter PUF  (k=1 — mathematically weak)"),
        (4, "XOR PUF k=4         (standard — moderate resistance)"),
        (8, "XOR PUF k=8         (strong — high resistance)"),
    ]

    for k, label in configs:
        print(f"\n\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        score = run_audit(
            n_crps     = args.crps,
            k_chains   = k,
            noise_rate = DEFAULT_NOISE_RATE,
            cv_folds   = 3,
            save_model = False,
            verbose    = True,
        )
        print(f"  → Reliability: {score.reliability_score:.4f}  "
              f"Verdict: {score.verdict.value}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost PUF Modeling Attack & Adversarial Auditor"
    )
    sub = parser.add_subparsers(dest="cmd")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--crps",   type=int,   default=5000,
                        help="Number of CRPs to collect (default 5000)")
    common.add_argument("--chains", type=int,   default=DEFAULT_XOR_CHAINS,
                        help=f"XOR chains k (default {DEFAULT_XOR_CHAINS})")
    common.add_argument("--noise",  type=float, default=DEFAULT_NOISE_RATE,
                        help=f"Noise rate (default {DEFAULT_NOISE_RATE})")
    common.add_argument("--cv",     type=int,   default=5,
                        help="CV folds (default 5)")

    sub.add_parser("attack", parents=[common],
                   help="Run modeling attack and report vulnerability")
    sub.add_parser("audit",  parents=[common],
                   help="Run audit and show history")
    demo_p = sub.add_parser("demo",
                             help="Compare secure vs vulnerable PUF configs")
    demo_p.add_argument("--crps", type=int, default=3000,
                        help="CRPs per config (default 3000)")

    args = parser.parse_args()

    if args.cmd == "attack":
        sys.exit(_cmd_attack(args))
    elif args.cmd == "audit":
        sys.exit(_cmd_audit(args))
    elif args.cmd == "demo":
        sys.exit(_cmd_demo(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
