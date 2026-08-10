# test_sca_clean.py
from sca_module import SCAMonitor, SVMSCADetector, INA219Sampler
import time

# 1. Initialize detector and load the existing model directly
print("Loading pre-trained model...")
detector = SVMSCADetector()
detector.load("sca_model.pkl")  # Loads the model you just trained on 1500 samples

# 2. Run a non-secret operation in the monitor
sampler = INA219Sampler()
print("\nStarting clean operation monitor...")
with SCAMonitor(detector=detector, sampler=sampler, min_monitor_seconds=2.0) as monitor:
    # Do something that doesn't depend on a secret
    for i in range(1000):
        x = i * 2  # constant-time work
        time.sleep(0.001) # Tiny sleep ensures we actually span the 2.0 seconds
        
# 3. Get the report
report = monitor.session_report()
print(f"\nClean operation: {report['leakage_fraction']:.1%} leakage")
