# sca_module.py — SVM-based Side-Channel Attack (SCA) Detection
# ============================================================
# Monitors power consumption via I2C (INA219 current sensor)
# while Kyber PQC operations run, then classifies each trace
# as NORMAL or LEAKAGE using a trained SVM.
#
# Hardware:  INA219 breakout board on I2C bus (default addr 0x40)
#            Raspberry Pi 5  →  SDA=GPIO2, SCL=GPIO3
#
# Pipeline:
#   1. Sample   — collect N current readings during a crypto op
#   2. Features — extract statistical + frequency features
#   3. Train    — fit SVM on labelled normal / leakage traces
#   4. Infer    — classify live traces; alert on leakage
#   5. Monitor  — background thread wraps steps 1+4 continuously
# ============================================================

from __future__ import annotations

import os
import time
import threading
import logging
import pickle
import struct
import hashlib
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Callable

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# ── Logging ────────────────────────────────────────────────
log = logging.getLogger("sca_module")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "[SCA %(levelname)s %(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    log.addHandler(_h)


# ============================================================
# Constants
# ============================================================

MODEL_PATH        = "sca_model.pkl"
DATASET_PATH      = "sca_dataset.npz"

# INA219 I2C defaults
INA219_ADDR       = 0x40
INA219_REG_SHUNT  = 0x01   # shunt voltage register
INA219_REG_BUS    = 0x02   # bus voltage register
INA219_REG_POWER  = 0x03   # power register
INA219_REG_CALIB  = 0x05   # calibration register

# Sampling
# INA219 via smbus2 on Pi 5 achieves ~100–200 reads/s due to I2C overhead
# (~5 ms per read). 200 Hz is the practical ceiling; the simulator honours
# this rate via time.sleep() so behaviour matches real hardware exactly.
SAMPLE_RATE_HZ      = 200    # samples per second (real INA219 I2C limit)
WINDOW_SAMPLES      = 20     # samples per window → 20/200 = 100 ms per window
FEATURE_DIM         = 18     # number of extracted features (see below)

# Minimum time the SCAMonitor must collect data for.
# Kyber keygen / encap / decap completes in ~2–5 ms, which is far shorter
# than one collection window. Without a minimum duration the monitor stops
# after a single window (giving 1 data point — statistically useless).
# 2 seconds → 2000 ms / 100 ms per window = ~20 windows per Phase 3 session.
MIN_MONITOR_SECONDS = 2.0

# SVM hyperparameters (tuned for this task)
SVM_C             = 10.0
SVM_GAMMA         = "scale"
SVM_KERNEL        = "rbf"

# Detection threshold: fraction of windows classified as LEAKAGE
# (over a 3-10 window smoothing buffer) before an alert fires.
# Used for longer sessions to avoid false positives from a single
# noisy/misclassified window.
ALERT_THRESHOLD   = 0.30

# Single-window override: if the SVM is at least this confident
# that ONE window is LEAKAGE, fire immediately instead of waiting
# for the smoothing buffer to fill. This matters because a short
# crypto op (e.g. one Kyber KeyGen call) may finish before 3
# windows can even be collected.
HIGH_CONFIDENCE_THRESHOLD = 0.90


# ============================================================
# Label enum
# ============================================================

class Label(int, Enum):
    NORMAL  = 0
    LEAKAGE = 1


# ============================================================
# Hardware abstraction — INA219 over smbus2
# Falls back to a deterministic simulator when no hardware found.
# ============================================================

class INA219Sampler:
    """
    Read instantaneous current (mA) from an INA219 over I2C.

    The INA219 shunt-voltage register gives a signed 16-bit value
    in units of 10 µV.  With a 0.1 Ω shunt: I = V_shunt / R_shunt.

    Typical Kyber key-gen draws ~180–220 mA at 3.3 V on a Pi 5.
    A side-channel leakage event shows sharp ±15–40 mA spikes
    correlated with NTT butterfly operations.
    """

    # Calibration for 32 V / 2 A range, 0.1 Ω shunt
    _CALIB_VALUE   = 4096
    _CURRENT_LSB   = 0.0001      # A per LSB  (0.1 mA resolution)
    _SHUNT_LSB_UV  = 10.0        # µV per LSB

    def __init__(self, address: int = INA219_ADDR, bus_num: int = 1):
        self.address  = address
        self.bus_num  = bus_num
        self._bus     = None
        self._sim     = False
        self._sim_rng = np.random.default_rng(seed=0)
        self._connect()

    # ── init ──────────────────────────────────────────────

    def _connect(self):
        try:
            import smbus2
            self._bus = smbus2.SMBus(self.bus_num)
            # Write calibration register
            self._write16(INA219_REG_CALIB, self._CALIB_VALUE)
            log.info("INA219 connected on bus %d addr 0x%02X",
                     self.bus_num, self.address)
        except Exception as exc:
            log.warning("INA219 not available (%s) — using simulator", exc)
            self._sim = True

    def _write16(self, reg: int, value: int):
        hi = (value >> 8) & 0xFF
        lo = value & 0xFF
        self._bus.write_i2c_block_data(self.address, reg, [hi, lo])

    def _read16_signed(self, reg: int) -> int:
        raw = self._bus.read_i2c_block_data(self.address, reg, 2)
        val = (raw[0] << 8) | raw[1]
        if val > 0x7FFF:
            val -= 0x10000
        return val

    # ── public API ────────────────────────────────────────

    def read_current_ma(self) -> float:
        """Return instantaneous current in mA."""
        if self._sim:
            return self._sim_current_ma()
        try:
            shunt_raw = self._read16_signed(INA219_REG_SHUNT)
            shunt_uv  = shunt_raw * self._SHUNT_LSB_UV
            current_a = shunt_uv / 1e6 / 0.1   # 0.1 Ω shunt
            return current_a * 1000.0           # → mA
        except Exception as exc:
            log.debug("INA219 read error: %s — simulating", exc)
            return self._sim_current_ma()

    def collect_window(self,
                       n_samples: int = WINDOW_SAMPLES,
                       rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
        """
        Collect n_samples readings at rate_hz.
        Returns shape (n_samples,) float32 array of mA values.
        """
        interval = 1.0 / rate_hz
        buf = np.empty(n_samples, dtype=np.float32)
        for i in range(n_samples):
            t0 = time.monotonic()
            buf[i] = self.read_current_ma()
            elapsed = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        return buf

    # ── simulator ─────────────────────────────────────────

    def _sim_current_ma(self) -> float:
        """
        Simulate Pi 5 power draw during Kyber operations.
        Base current ~200 mA with small Gaussian noise.
        Call sim_inject_leakage() to superimpose a leakage spike.
        """
        return float(
            200.0
            + self._sim_rng.normal(0, 2.5)
            + self._sim_base_delta
        )

    _sim_base_delta: float = 0.0     # injected by sim_inject_leakage()

    def sim_inject_leakage(self, active: bool = True):
        """Toggle simulated leakage spike (test harness only)."""
        if active:
            # sharp spike: +30 mA base + high-freq ripple seed
            self._sim_base_delta = 30.0
            self._sim_rng = np.random.default_rng(seed=42)
        else:
            self._sim_base_delta = 0.0
            self._sim_rng = np.random.default_rng(seed=0)

    def close(self):
        if self._bus:
            try:
                self._bus.close()
            except Exception:
                pass


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(trace: np.ndarray) -> np.ndarray:
    """
    Extract FEATURE_DIM = 18 statistical + frequency features
    from a raw current trace (1-D array, units: mA).

    Feature vector layout:
      [0]  mean
      [1]  std
      [2]  min
      [3]  max
      [4]  peak-to-peak (max - min)
      [5]  RMS
      [6]  skewness
      [7]  kurtosis
      [8]  zero-crossing rate
      [9]  energy  (sum of squares, normalised)
      [10] 25th percentile
      [11] 75th percentile
      [12] IQR
      [13-17] top-5 FFT magnitude bins (DC excluded)

    These features capture:
    - mean / std / RMS   → average power level and spread
    - skewness / kurtosis → asymmetric spike shapes
    - peak-to-peak       → sharp voltage transients
    - ZCR                → high-frequency switching noise
    - FFT bins           → frequency-domain leakage harmonics
      (Kyber NTT leaks at specific clock-frequency multiples)
    """
    t = trace.astype(np.float64)
    n = len(t)

    mean   = np.mean(t)
    std    = np.std(t)
    mn     = np.min(t)
    mx     = np.max(t)
    ptp    = mx - mn
    rms    = np.sqrt(np.mean(t ** 2))

    # Skewness  (3rd standardised moment)
    skew   = np.mean(((t - mean) / (std + 1e-9)) ** 3)
    # Excess kurtosis  (4th standardised moment − 3)
    kurt   = np.mean(((t - mean) / (std + 1e-9)) ** 4) - 3.0

    # Zero-crossing rate (fraction of adjacent pairs that change sign
    # relative to mean-subtracted signal)
    centred  = t - mean
    zcr      = np.sum(np.diff(np.sign(centred)) != 0) / max(n - 1, 1)

    # Normalised energy
    energy   = np.sum(t ** 2) / n

    # Percentiles
    p25, p75 = np.percentile(t, [25, 75])
    iqr      = p75 - p25

    # FFT — take magnitude of positive frequencies, skip DC (bin 0)
    fft_mag  = np.abs(np.fft.rfft(t - mean))[1:]          # skip DC
    # Pick top 5 bins by magnitude to capture dominant harmonics
    top_idx  = np.argsort(fft_mag)[-5:][::-1]
    fft_feats = fft_mag[top_idx] / (n + 1e-9)             # normalise

    feats = np.array([
        mean, std, mn, mx, ptp, rms,
        skew, kurt, zcr, energy,
        p25, p75, iqr,
        *fft_feats                                         # 5 values
    ], dtype=np.float32)

    assert len(feats) == FEATURE_DIM, \
        f"Feature dim mismatch: {len(feats)} != {FEATURE_DIM}"
    return feats


# ============================================================
# Dataset helpers
# ============================================================

@dataclass
class TraceDataset:
    """Labelled collection of power traces and their feature vectors."""
    traces:   List[np.ndarray] = field(default_factory=list)
    features: List[np.ndarray] = field(default_factory=list)
    labels:   List[int]        = field(default_factory=list)

    def add(self, trace: np.ndarray, label: Label):
        self.traces.append(trace)
        self.features.append(extract_features(trace))
        self.labels.append(int(label))

    def X(self) -> np.ndarray:
        return np.vstack(self.features)

    def y(self) -> np.ndarray:
        return np.array(self.labels, dtype=np.int32)

    def save(self, path: str = DATASET_PATH):
        np.savez_compressed(
            path,
            features=self.X(),
            labels=self.y()
        )
        log.info("Dataset saved → %s  (%d samples)", path, len(self.labels))

    @classmethod
    def load(cls, path: str = DATASET_PATH) -> "TraceDataset":
        data = np.load(path)
        ds = cls()
        for feat, lbl in zip(data["features"], data["labels"]):
            ds.features.append(feat)
            ds.labels.append(int(lbl))
            ds.traces.append(np.array([]))   # raw trace not stored
        log.info("Dataset loaded ← %s  (%d samples)", path, len(ds.labels))
        return ds


# ============================================================
# Synthetic dataset generator (used when real hardware absent)
# ============================================================

def generate_synthetic_dataset(
        n_normal: int = 800,
        n_leakage: int = 200,
        n_samples: int = WINDOW_SAMPLES,
        rng_seed: int = 0
) -> TraceDataset:
    """
    Generate labelled synthetic power traces that mimic real
    Pi 5 / INA219 measurements during Kyber operations.

    Normal trace:
      ~200 mA baseline  +  σ=3 mA Gaussian noise
      +  low-amplitude sinusoid (switching regulator ripple, ~50 Hz)

    Leakage trace:
      Same baseline  +  one or more sharp current spikes (±20–50 mA)
      correlated with NTT butterfly indices.  The spikes have a
      characteristic shape (fast rise, exponential decay) that the
      SVM learns to distinguish from Gaussian noise.
    """
    rng = np.random.default_rng(rng_seed)
    ds  = TraceDataset()
    t   = np.linspace(0, n_samples / SAMPLE_RATE_HZ, n_samples)

    # ── Normal traces ──────────────────────────────────────
    for _ in range(n_normal):
        baseline  = 200.0 + rng.normal(0, 0.5)
        noise     = rng.normal(0, 3.0, n_samples)
        ripple    = 1.5 * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 2*np.pi))
        trace     = (baseline + noise + ripple).astype(np.float32)
        ds.add(trace, Label.NORMAL)

    # ── Leakage traces ─────────────────────────────────────
    for _ in range(n_leakage):
        baseline  = 200.0 + rng.normal(0, 0.5)
        noise     = rng.normal(0, 3.0, n_samples)
        ripple    = 1.5 * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 2*np.pi))

        # Inject 2–5 NTT-style current spikes
        n_spikes  = rng.integers(2, 6)
        spike_pos = rng.integers(5, n_samples - 5, size=n_spikes)
        spike_amp = rng.uniform(20, 50, size=n_spikes) * rng.choice([-1, 1], n_spikes)

        spike_sig = np.zeros(n_samples)
        for pos, amp in zip(spike_pos, spike_amp):
            # Fast rise (1 sample), exponential decay (τ ≈ 3 samples)
            for j in range(n_samples - pos):
                spike_sig[pos + j] += amp * np.exp(-j / 3.0)

        trace = (baseline + noise + ripple + spike_sig).astype(np.float32)
        ds.add(trace, Label.LEAKAGE)

    log.info("Synthetic dataset: %d normal + %d leakage traces",
             n_normal, n_leakage)
    return ds


# ============================================================
# SVM Model  (sklearn Pipeline: StandardScaler → RBF-SVM)
# ============================================================

class SVMSCADetector:
    """
    Binary SVM classifier wrapping an sklearn Pipeline.

    Predict a single feature vector → Label (NORMAL / LEAKAGE).
    Also exposes a decision_score for confidence reporting.
    """

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._trained: bool = False

    # ── Training ──────────────────────────────────────────

    def train(self, dataset: TraceDataset, cv_folds: int = 5) -> dict:
        """
        Fit the SVM on the dataset.  Runs stratified k-fold CV
        to report generalisation metrics before returning.

        Returns a dict with keys: accuracy, precision, recall, f1
        """
        X, y = dataset.X(), dataset.y()
        log.info("Training SVM on %d samples (dim=%d)…", len(y), X.shape[1])

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(
                C=SVM_C,
                kernel=SVM_KERNEL,
                gamma=SVM_GAMMA,
                class_weight="balanced",   # handles imbalanced datasets
                probability=True,           # enables predict_proba
                random_state=42,
                cache_size=500,
            ))
        ])

        # Cross-validation (runs on CPU only — fits Pi 5 well)
        skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self._pipeline, X, y,
                                 cv=skf, scoring="f1", n_jobs=-1)
        log.info("CV F1: %.3f ± %.3f", scores.mean(), scores.std())

        # Final fit on the full training set
        self._pipeline.fit(X, y)
        self._trained = True

        # In-sample report (for sanity; CV scores are the real metric)
        y_pred  = self._pipeline.predict(X)
        report  = classification_report(y, y_pred, target_names=["NORMAL", "LEAKAGE"],
                                        output_dict=True, zero_division=0)
        cm      = confusion_matrix(y, y_pred)

        log.info("Confusion matrix (train set):\n%s", cm)
        log.info("Per-class report:\n%s",
                 classification_report(y, y_pred,
                                       target_names=["NORMAL", "LEAKAGE"],
                                       zero_division=0))

        metrics = {
            "cv_f1_mean":  float(scores.mean()),
            "cv_f1_std":   float(scores.std()),
            "accuracy":    float(report["accuracy"]),
            "precision":   float(report["LEAKAGE"]["precision"]),
            "recall":      float(report["LEAKAGE"]["recall"]),
            "f1":          float(report["LEAKAGE"]["f1-score"]),
            "confusion_matrix": cm.tolist(),
        }
        return metrics

    # ── Inference ─────────────────────────────────────────

    def predict(self, trace: np.ndarray) -> Tuple[Label, float]:
        """
        Classify one raw trace.

        Returns: (Label.NORMAL or Label.LEAKAGE, confidence 0–1)
        """
        self._require_trained()
        feat  = extract_features(trace).reshape(1, -1)
        label = int(self._pipeline.predict(feat)[0])
        proba = self._pipeline.predict_proba(feat)[0][label]
        return Label(label), float(proba)

    def decision_score(self, trace: np.ndarray) -> float:
        """
        Raw SVM decision function value.
        Positive → LEAKAGE side, negative → NORMAL side.
        Useful for threshold tuning.
        """
        self._require_trained()
        feat = extract_features(trace).reshape(1, -1)
        return float(self._pipeline.decision_function(feat)[0])

    # ── Persistence ───────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        self._require_trained()
        with open(path, "wb") as f:
            pickle.dump(self._pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Model saved → %s", path)

    def load(self, path: str = MODEL_PATH):
        with open(path, "rb") as f:
            self._pipeline = pickle.load(f)
        self._trained = True
        log.info("Model loaded ← %s", path)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _require_trained(self):
        if not self._trained:
            raise RuntimeError(
                "SVMSCADetector: model not trained. "
                "Call train() or load() first."
            )


# ============================================================
# Alert / Event System
# ============================================================

@dataclass
class SCAEvent:
    """Fired when a leakage pattern is detected."""
    timestamp:    float          # time.time()
    confidence:   float          # SVM probability for LEAKAGE class
    score:        float          # raw decision function value
    window_index: int            # which window in the session
    trace_hash:   str            # SHA-256 of the raw trace (for logging)

    def __str__(self):
        return (
            f"⚠  SCA LEAKAGE DETECTED  "
            f"[window={self.window_index}  "
            f"conf={self.confidence:.1%}  "
            f"score={self.score:+.3f}  "
            f"trace={self.trace_hash[:12]}…]"
        )


AlertCallback = Callable[[SCAEvent], None]


def _default_alert(event: SCAEvent):
    """Default: print a red alert and log at WARNING level."""
    RED  = "\033[91m"
    BOLD = "\033[1m"
    END  = "\033[0m"
    print(f"\n{RED}{BOLD}{event}{END}\n", flush=True)
    log.warning(str(event))


# ============================================================
# Background Monitor Thread
# ============================================================

class SCAMonitor:
    """
    Runs in a daemon thread; continuously samples power, classifies
    windows, and fires alert callbacks when leakage is detected.

    Usage:
        detector = SVMSCADetector()
        detector.load()   # or .train(dataset)

        monitor = SCAMonitor(detector)
        monitor.start()

        # … run PQC operations …

        monitor.stop()
        report = monitor.session_report()
    """

    def __init__(self,
                 detector:            SVMSCADetector,
                 sampler:             Optional[INA219Sampler] = None,
                 alert_callback:      AlertCallback = _default_alert,
                 window_samples:      int   = WINDOW_SAMPLES,
                 sample_rate_hz:      int   = SAMPLE_RATE_HZ,
                 alert_threshold:     float = ALERT_THRESHOLD,
                 min_monitor_seconds: float = MIN_MONITOR_SECONDS):
        """
        min_monitor_seconds:
            The monitor will NOT stop until at least this many seconds
            have elapsed since start(), regardless of when the crypto
            operation finishes.  This ensures enough windows are
            collected even when the PQC operation completes in <5 ms.

            At WINDOW_SAMPLES=20, SAMPLE_RATE_HZ=200:
              each window takes 100 ms → 2.0 s gives ~20 windows.
        """
        self._detector           = detector
        self._sampler            = sampler or INA219Sampler()
        self._alert_cb           = alert_callback
        self._window_samples     = window_samples
        self._rate_hz            = sample_rate_hz
        self._alert_threshold    = alert_threshold
        self._min_monitor_secs   = min_monitor_seconds

        self._thread:             Optional[threading.Thread] = None
        self._stop_event:         threading.Event = threading.Event()
        self._started_at:         float = 0.0      # set in start()

        # Session statistics
        self._lock:               threading.Lock = threading.Lock()
        self._total_windows:      int = 0
        self._leakage_windows:    int = 0
        self._events:             List[SCAEvent] = []
        self._active:             bool = False

    # ── Lifecycle ─────────────────────────────────────────

    def start(self):
        """Start the background monitoring thread."""
        if self._active:
            return
        self._stop_event.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name="SCAMonitor",
            daemon=True
        )
        self._active = True
        self._thread.start()
        log.info("SCA monitor started (window=%d samples @ %d Hz, min=%.1fs)",
                 self._window_samples, self._rate_hz, self._min_monitor_secs)

    def stop(self, timeout: float = 10.0):
        """Signal the monitoring thread to stop and join it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._active = False
        log.info("SCA monitor stopped. "
                 "Windows: %d total / %d leakage (%.1f%%)",
                 self._total_windows,
                 self._leakage_windows,
                 100 * self._leakage_windows / max(self._total_windows, 1))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        # ── Minimum-duration enforcement ──────────────────────
        # The PQC operation (keygen + decapsulation) completes in
        # ~2–5 ms, far faster than one 100 ms collection window.
        # Without this wait the monitor stops after just 1 window.
        # We hold here until min_monitor_seconds have elapsed so
        # the background thread collects enough windows (~20) to
        # give the SVM a statistically meaningful assessment.
        if self._min_monitor_secs > 0:
            elapsed  = time.monotonic() - self._started_at
            remaining = self._min_monitor_secs - elapsed
            if remaining > 0:
                log.info(
                    "SCA min-duration pad: waiting %.2f s for more windows...",
                    remaining,
                )
                time.sleep(remaining)
        self.stop()

    # ── Main loop ─────────────────────────────────────────

    def _run(self):
        window_idx = 0
        recent_labels: List[int] = []   # sliding window for threshold check

        while not self._stop_event.is_set():
            # 1. Collect one power trace window
            try:
                trace = self._sampler.collect_window(
                    n_samples=self._window_samples,
                    rate_hz=self._rate_hz
                )
            except Exception as exc:
                log.debug("Sampler error: %s", exc)
                time.sleep(0.05)
                continue

            # 2. Classify
            try:
                label, confidence = self._detector.predict(trace)
                score             = self._detector.decision_score(trace)
            except Exception as exc:
                log.debug("Prediction error: %s", exc)
                window_idx += 1
                continue

            # 3. Update session stats
            with self._lock:
                self._total_windows += 1
                if label == Label.LEAKAGE:
                    self._leakage_windows += 1

            recent_labels.append(int(label))
            if len(recent_labels) > 10:
                recent_labels.pop(0)

            window_idx += 1

            # 4. Decide whether to fire an alert.
            #
            # Two paths, because crypto ops vary wildly in duration:
            #
            #  (a) HIGH-CONFIDENCE SINGLE-SHOT — a short operation
            #      (e.g. one Kyber KeyGen call) may only produce 1-2
            #      windows total before the op finishes and the monitor
            #      is stopped. If the model is very confident about that
            #      one window, don't wait for a 3-window smoothing
            #      buffer that will never fill in time — fire now.
            #
            #  (b) SMOOTHED ROLLING THRESHOLD — for longer sessions
            #      (idle monitoring, repeated ops, full chat phases)
            #      require >= 3 windows and a sustained leakage
            #      fraction so a single noisy/misclassified window
            #      doesn't trigger a false alarm.
            fire = False
            if label == Label.LEAKAGE and confidence >= HIGH_CONFIDENCE_THRESHOLD:
                fire = True
            elif len(recent_labels) >= 3:
                leakage_frac = sum(recent_labels) / len(recent_labels)
                if leakage_frac >= self._alert_threshold:
                    fire = True

            if fire:
                trace_hash = hashlib.sha256(
                    trace.tobytes()
                ).hexdigest()
                event = SCAEvent(
                    timestamp=time.time(),
                    confidence=confidence,
                    score=score,
                    window_index=window_idx,
                    trace_hash=trace_hash,
                )
                with self._lock:
                    self._events.append(event)
                try:
                    self._alert_cb(event)
                except Exception as exc:
                    log.debug("Alert callback error: %s", exc)
                # Reset recent window to avoid immediately re-firing
                # on the same sustained condition.
                recent_labels.clear()

    # ── Session report ────────────────────────────────────

    def session_report(self) -> dict:
        """Return a summary dict for the completed session."""
        with self._lock:
            total    = self._total_windows
            leakage  = self._leakage_windows
            events   = list(self._events)

        return {
            "total_windows":    total,
            "leakage_windows":  leakage,
            "leakage_fraction": leakage / max(total, 1),
            "alert_count":      len(events),
            "events":           [str(e) for e in events],
            "clean_session":    len(events) == 0,
        }


# ============================================================
# Convenience: wrap a PQC operation with SCA monitoring
# ============================================================

def monitored_pqc_op(
        op_callable: Callable,
        detector:    SVMSCADetector,
        sampler:     Optional[INA219Sampler] = None,
        label:       str = "pqc_op"
) -> Tuple[any, dict]:
    """
    Run `op_callable()` while the SCA monitor watches power consumption.

    Returns: (op_result, session_report_dict)

    Example:
        result, report = monitored_pqc_op(
            lambda: pqc.generate_keypair(),
            detector=svm_detector
        )
    """
    alerts_fired = []

    def on_alert(event: SCAEvent):
        _default_alert(event)
        alerts_fired.append(event)

    monitor = SCAMonitor(
        detector=detector,
        sampler=sampler,
        alert_callback=on_alert,
    )

    monitor.start()
    try:
        result = op_callable()
    finally:
        monitor.stop()

    report = monitor.session_report()
    report["op_label"] = label

    if report["clean_session"]:
        log.info("[%s] Clean execution — no SCA leakage detected.", label)
    else:
        log.warning("[%s] %d alert(s) fired during execution!",
                    label, report["alert_count"])

    return result, report


# ============================================================
# Training pipeline (called once during setup)
# ============================================================

def train_sca_model(
        dataset:        Optional[TraceDataset] = None,
        save_model_path: str = MODEL_PATH,
        save_data_path:  str = DATASET_PATH,
) -> Tuple[SVMSCADetector, dict]:
    """
    End-to-end training pipeline:
      1. If no dataset supplied, generate synthetic one.
      2. Save dataset to disk.
      3. Train SVM, report CV metrics.
      4. Save model to disk.

    Returns: (trained_detector, metrics_dict)
    """
    if dataset is None:
        log.info("No dataset provided — generating synthetic data...")
        dataset = generate_synthetic_dataset()

    dataset.save(save_data_path)

    detector = SVMSCADetector()
    metrics  = detector.train(dataset)
    detector.save(save_model_path)

    log.info("Training complete.")
    log.info("  CV F1:     %.3f ± %.3f", metrics["cv_f1_mean"], metrics["cv_f1_std"])
    log.info("  Accuracy:  %.3f",         metrics["accuracy"])
    log.info("  Precision: %.3f",         metrics["precision"])
    log.info("  Recall:    %.3f",         metrics["recall"])

    return detector, metrics


# ============================================================
# CLI / self-test
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCA Module — train or test")
    parser.add_argument("mode", choices=["train", "test", "demo"],
                        help="train: fit and save model | "
                             "test: load and evaluate | "
                             "demo: full end-to-end demo")
    args = parser.parse_args()

    if args.mode == "train":
        print("\n=== SCA MODEL TRAINING ===")
        detector, metrics = train_sca_model()
        print(f"\n  CV F1:     {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"\n  Model saved to: {MODEL_PATH}")

    elif args.mode == "test":
        print("\n=== SCA MODEL EVALUATION ===")
        detector = SVMSCADetector()
        detector.load(MODEL_PATH)

        sampler = INA219Sampler()

        print("Collecting 20 test windows (10 normal / 10 leakage)...")
        correct = 0
        for i in range(20):
            if i < 10:
                sampler.sim_inject_leakage(False)
                true_label = Label.NORMAL
            else:
                sampler.sim_inject_leakage(True)
                true_label = Label.LEAKAGE

            trace = sampler.collect_window(n_samples=WINDOW_SAMPLES)
            pred, conf = detector.predict(trace)
            ok = pred == true_label
            correct += ok
            sym = "✓" if ok else "✗"
            print(f"  [{sym}] Window {i+1:2d}  true={true_label.name:7s}  "
                  f"pred={pred.name:7s}  conf={conf:.1%}")

        sampler.sim_inject_leakage(False)
        sampler.close()
        print(f"\n  Accuracy on test set: {correct}/20 ({100*correct/20:.0f}%)")

    elif args.mode == "demo":
        print("\n=== SCA FULL DEMO ===")

        # 1. Train
        print("\n[1] Training model on synthetic data…")
        detector, metrics = train_sca_model()
        print(f"    CV F1: {metrics['cv_f1_mean']:.3f}")

        # 2. Simulate a PQC key-gen under monitoring
        print("\n[2] Simulating Kyber key-gen with SCA monitoring…")
        sampler = INA219Sampler()

        def fake_kyber_keygen():
            """Simulate ~200 ms key-gen with injected leakage partway through."""
            time.sleep(0.05)
            sampler.sim_inject_leakage(True)   # spike halfway
            time.sleep(0.10)
            sampler.sim_inject_leakage(False)
            time.sleep(0.05)
            return b"fake_public_key_bytes"

        result, report = monitored_pqc_op(
            fake_kyber_keygen,
            detector=detector,
            sampler=sampler,
            label="Kyber-KeyGen"
        )

        print("\n[3] Session report:")
        for k, v in report.items():
            if k != "events":
                print(f"    {k}: {v}")
        if report["events"]:
            print("    Events:")
            for e in report["events"]:
                print(f"      {e}")

        sampler.close()
        print("\n  Demo complete.")
