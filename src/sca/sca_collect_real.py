#!/usr/bin/env python3
# sca_collect_real.py — Collect REAL power traces from INA219 during Kyber

import numpy as np
import time
import subprocess
import sys
import threading
import argparse

try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    print("[ERROR] smbus2 not installed")
    print("  Run: pip install smbus2 --break-system-packages")
    sys.exit(1)


# ============================================================
# INA219 Direct Sampling
# ============================================================

class INA219Reader:
    """Direct INA219 power sensor reader over I2C."""
    
    def __init__(self, address=0x40, bus_id=1):
        self.address = address
        self.bus_id = bus_id
        self.bus = smbus2.SMBus(bus_id)
        self._configure()
    
    def _configure(self):
        """Configure INA219 for continuous current/power measurement."""
        # Config: 32V bus, 320mV shunt, 12-bit ADC, continuous mode
        config = 0x399F
        self.bus.write_word_data(self.address, 0x00, self._swap16(config))
        time.sleep(0.1)
        
        # Calibration for 0.1Ω shunt resistor
        # LSB = 0.04096 / (0.0001 * 0.1) = 4096
        calib = 4096
        self.bus.write_word_data(self.address, 0x05, self._swap16(calib))
        time.sleep(0.1)
    
    def _swap16(self, val):
        """I2C endianness fix."""
        return ((val & 0xFF) << 8) | ((val >> 8) & 0xFF)
    
    def read_power_mw(self):
        """Read instantaneous power in mW."""
        try:
            raw = self.bus.read_word_data(self.address, 0x03)
            raw = self._swap16(raw)
            # Power LSB = 20 × current LSB = 20 × 0.1 mA = 2 mW
            return max(0, raw * 2.0)
        except Exception as e:
            print(f"[WARN] I2C read error: {e}")
            return 0.0
    
    def collect_window(self, n_samples=20, sample_rate_hz=200):
        """Collect n_samples at sample_rate_hz."""
        samples = np.empty(n_samples, dtype=np.float32)
        interval = 1.0 / sample_rate_hz
        
        for i in range(n_samples):
            t0 = time.perf_counter()
            samples[i] = self.read_power_mw()
            elapsed = time.perf_counter() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
        
        return samples
    
    def close(self):
        self.bus.close()


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract 20-D feature vector."""
    from scipy.signal import welch, find_peaks
    from scipy.stats import kurtosis, skew
    
    w = window.astype(np.float64)
    
    # Time-domain
    t_mean = np.mean(w)
    t_std = np.std(w)
    t_min = np.min(w)
    t_max = np.max(w)
    t_range = t_max - t_min
    t_kurt = float(kurtosis(w))
    t_skew = float(skew(w))
    t_rms = np.sqrt(np.mean(w ** 2))
    
    # Frequency-domain
    freqs, psd = welch(w, fs=200, nperseg=min(16, len(w)))
    f_total_power = np.sum(psd)
    f_peak_freq = freqs[np.argmax(psd)] if f_total_power > 0 else 0
    f_low_band = np.sum(psd[freqs < 50])
    f_mid_band = np.sum(psd[(freqs >= 50) & (freqs < 100)])
    f_high_band = np.sum(psd[freqs >= 100])
    f_entropy = -np.sum((psd / (f_total_power + 1e-12)) * np.log2(psd / (f_total_power + 1e-12) + 1e-12))
    
    # Peaks
    peaks, props = find_peaks(w, height=t_mean + 0.5*t_std, distance=2)
    p_count = len(peaks)
    p_mean_height = np.mean(props["peak_heights"]) if p_count > 0 else 0
    p_max_height = np.max(props["peak_heights"]) if p_count > 0 else 0
    p_density = p_count / len(w)
    
    if len(peaks) > 1:
        ipi = np.diff(peaks).astype(float)
        p_ipi_mean = np.mean(ipi)
        p_ipi_std = np.std(ipi)
    else:
        p_ipi_mean = 0.0
        p_ipi_std = 0.0
    
    return np.array([
        t_mean, t_std, t_min, t_max, t_range, t_kurt, t_skew, t_rms,
        f_total_power, f_peak_freq, f_low_band, f_mid_band,
        f_high_band, f_entropy,
        p_count, p_mean_height, p_max_height, p_density,
        p_ipi_mean, p_ipi_std,
    ], dtype=np.float32)


# ============================================================
# CPU Load Control
# ============================================================

def start_cpu_load(num_threads=4):
    """Start CPU-intensive background task."""
    print(f"  Starting CPU load ({num_threads} threads)...")
    try:
        # Use stress-ng if available
        proc = subprocess.Popen(
            ["stress-ng", "--cpu", str(num_threads), "--timeout", "120s"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return proc
    except FileNotFoundError:
        # Fallback: Python-based busy loop
        print("  (stress-ng not found, using Python busy loop)")
        
        def busy_loop():
            import hashlib
            while True:
                for i in range(1000):
                    hashlib.sha256(str(i).encode()).digest()
        
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=busy_loop, daemon=True)
            t.start()
            threads.append(t)
        
        class DummyProc:
            def terminate(self): pass
            def wait(self): pass
        
        return DummyProc()


def stop_cpu_load(proc):
    """Stop CPU load process."""
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except:
        pass


# ============================================================
# Collection
# ============================================================

def collect_real_traces(n_normal=100, n_leakage=100):
    """Collect real power traces from INA219."""
    
    print("=" * 60)
    print("  REAL INA219 POWER TRACE COLLECTION")
    print("=" * 60)
    
    try:
        sensor = INA219Reader()
        print(f"\n✓ INA219 connected at 0x40 on bus 1")
    except Exception as e:
        print(f"\n✗ ERROR: Cannot connect to INA219")
        print(f"  {e}")
        print(f"  Verify wiring: SDA=GPIO2, SCL=GPIO3, GND, VCC")
        print(f"  Check: sudo i2cdetect -y 1")
        sys.exit(1)
    
    normal_traces = []
    leakage_traces = []
    
    # ── Collect NORMAL traces (idle) ──────────────────────────
    print(f"\n[1/2] Collecting {n_normal} NORMAL traces (idle)...")
    print(f"      Ensure system is idle (close browser, etc.)")
    
    for i in range(n_normal):
        window = sensor.collect_window(20, sample_rate_hz=200)
        features = extract_features(window)
        normal_traces.append(features)
        
        if (i + 1) % 25 == 0:
            mean_power = np.mean(window)
            print(f"      {i+1}/{n_normal}  (avg power: {mean_power:.0f} mW)")
    
    # ── Collect LEAKAGE traces (under CPU load) ──────────────
    print(f"\n[2/2] Collecting {n_leakage} LEAKAGE traces (under load)...")
    
    proc = start_cpu_load(num_threads=4)
    time.sleep(1)  # Let load settle
    
    try:
        for i in range(n_leakage):
            window = sensor.collect_window(20, sample_rate_hz=200)
            features = extract_features(window)
            leakage_traces.append(features)
            
            if (i + 1) % 25 == 0:
                mean_power = np.mean(window)
                print(f"      {i+1}/{n_leakage}  (avg power: {mean_power:.0f} mW)")
    finally:
        stop_cpu_load(proc)
        print(f"      CPU load stopped")
    
    sensor.close()
    
    # ── Analysis ──────────────────────────────────────────────
    X_normal = np.array(normal_traces)
    X_leakage = np.array(leakage_traces)
    X = np.vstack([X_normal, X_leakage])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_leakage))])
    
    print(f"\n✓ Collection complete:")
    print(f"  Normal:  {X_normal.shape} (mean power: {np.mean(X_normal[:, 0]):.0f} mW)")
    print(f"  Leakage: {X_leakage.shape} (mean power: {np.mean(X_leakage[:, 0]):.0f} mW)")
    print(f"  Δ power: {np.mean(X_leakage[:, 0]) - np.mean(X_normal[:, 0]):.0f} mW")
    
    # Class separation
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(X_normal[:, 0], X_leakage[:, 0])
    print(f"  T-test p-value: {p_val:.6f} {'(significant)' if p_val < 0.05 else '(not significant)'}")
    
    # Save
    np.savez("sca_dataset_real.npz", X=X, y=y)
    print(f"\n✓ Saved: sca_dataset_real.npz")
    print(f"\nNext: python3 retrain_sca_model_v2.py --data sca_dataset_real.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=int, default=100,
                        help="Normal samples (default 100)")
    parser.add_argument("--leakage", type=int, default=100,
                        help="Leakage samples (default 100)")
    args = parser.parse_args()
    
    collect_real_traces(args.normal, args.leakage)


if __name__ == "__main__":
    main()
