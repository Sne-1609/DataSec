#!/usr/bin/env python3
# diagnose_traces.py — Analyze if normal vs leakage traces are actually different

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("=" * 60)
print("  TRACE ANALYSIS")
print("=" * 60)

# Load
X_normal = np.load("normal_traces.npy")
X_leakage = np.load("leakage_traces.npy")

print(f"\nNormal traces:  {X_normal.shape}")
print(f"Leakage traces: {X_leakage.shape}")

# Feature-by-feature analysis
print("\n" + "=" * 60)
print("  FEATURE STATISTICS")
print("=" * 60)

for feat_idx in range(min(5, X_normal.shape[1])):  # Show first 5 features
    normal_vals = X_normal[:, feat_idx]
    leakage_vals = X_leakage[:, feat_idx]
    
    # T-test
    t_stat, p_value = stats.ttest_ind(normal_vals, leakage_vals)
    
    print(f"\nFeature {feat_idx}:")
    print(f"  Normal:  mean={np.mean(normal_vals):.2f} ± {np.std(normal_vals):.2f}")
    print(f"  Leakage: mean={np.mean(leakage_vals):.2f} ± {np.std(leakage_vals):.2f}")
    print(f"  T-test p-value: {p_value:.6f} {'***' if p_value < 0.05 else '(not significant)'}")
    
    # Effect size (Cohen's d)
    d = (np.mean(leakage_vals) - np.mean(normal_vals)) / np.sqrt((np.std(normal_vals)**2 + np.std(leakage_vals)**2) / 2)
    print(f"  Cohen's d (effect size): {d:.3f}")

# Overall class separability
print("\n" + "=" * 60)
print("  CLASS SEPARABILITY")
print("=" * 60)

normal_mean = np.mean(X_normal, axis=0)
leakage_mean = np.mean(X_leakage, axis=0)
euclidean_dist = np.linalg.norm(normal_mean - leakage_mean)

print(f"Euclidean distance between class centroids: {euclidean_dist:.3f}")

if euclidean_dist < 1.0:
    print("  ⚠ WARNING: Classes are VERY CLOSE (poor separability)")
    print("  This explains 50% accuracy — model can't learn the difference")
elif euclidean_dist < 5.0:
    print("  ⚠ WARNING: Classes are somewhat overlapped")
elif euclidean_dist > 10.0:
    print("  ✓ Classes are well-separated")
else:
    print("  ~ Classes have moderate separation")

# Visualize
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Feature 0 distribution
ax = axes[0, 0]
ax.hist(X_normal[:, 0], bins=20, alpha=0.6, label='Normal')
ax.hist(X_leakage[:, 0], bins=20, alpha=0.6, label='Leakage')
ax.set_xlabel('Feature 0 (Mean Power)')
ax.set_ylabel('Count')
ax.legend()
ax.set_title('Feature 0 Distribution')

# Plot 2: Feature 1 distribution
ax = axes[0, 1]
ax.hist(X_normal[:, 1], bins=20, alpha=0.6, label='Normal')
ax.hist(X_leakage[:, 1], bins=20, alpha=0.6, label='Leakage')
ax.set_xlabel('Feature 1 (Std Dev)')
ax.set_ylabel('Count')
ax.legend()
ax.set_title('Feature 1 Distribution')

# Plot 3: Scatter (Feature 0 vs 1)
ax = axes[1, 0]
ax.scatter(X_normal[:, 0], X_normal[:, 1], alpha=0.6, label='Normal')
ax.scatter(X_leakage[:, 0], X_leakage[:, 1], alpha=0.6, label='Leakage')
ax.set_xlabel('Feature 0 (Mean)')
ax.set_ylabel('Feature 1 (Std Dev)')
ax.legend()
ax.set_title('Feature Space')

# Plot 4: Class means
ax = axes[1, 1]
features_to_plot = range(min(10, X_normal.shape[1]))
ax.plot(features_to_plot, normal_mean[:10], 'o-', label='Normal', linewidth=2)
ax.plot(features_to_plot, leakage_mean[:10], 's-', label='Leakage', linewidth=2)
ax.set_xlabel('Feature Index')
ax.set_ylabel('Mean Value')
ax.legend()
ax.set_title('Feature Means Comparison')

plt.tight_layout()
plt.savefig('trace_analysis.png', dpi=100)
print("✓ Saved: trace_analysis.png")

# Diagnosis
print("\n" + "=" * 60)
print("  DIAGNOSIS")
print("=" * 60)

if euclidean_dist < 1.0:
    print("""
✗ PROBLEM: Normal and leakage traces are nearly IDENTICAL
  
  Root causes:
  1. CPU load didn't actually change power consumption
     → Kyber/crypto ops are too fast to see under load
     → I2C sampling is too slow to catch transients
  
  2. INA219 resolution is insufficient
     → 2 mW LSB may be too coarse
     → Need higher-resolution ADC or external amplifier
  
  3. Collection conditions were poor
     → System still had other tasks running (browser, daemons)
     → Need to minimize background activity
  
  SOLUTIONS:
  - Run `sudo systemctl isolate multi-user.target` to kill GUI
  - Use `stress-ng --cpu 4 --timeout 60s` for stronger load
  - Collect longer sequences (120 traces each, not 60)
  - Try `python3 simple_sca_test.py` to verify INA219 responds to load
""")
elif euclidean_dist < 5.0:
    print("""
~ MARGINAL: Classes have some separation but overlapped
  
  The signal exists but is noisy. Try:
  1. Increase sample count: collect 200 traces each (not 60)
  2. Increase CPU load intensity
  3. Apply bandpass filtering to remove noise
""")
else:
    print("""
✓ GOOD: Classes are well-separated
  
  If model still shows 50% accuracy, the problem is in training:
  1. Hyperparameters: try C=0.1 or C=100 instead of C=10
  2. Feature scaling: verify StandardScaler is working
  3. Train/test split: try different random_state values
""")
