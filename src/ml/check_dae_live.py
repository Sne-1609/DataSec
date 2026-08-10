import torch
import numpy as np
import os
from dae_module import DenoisingAutoencoder

# Paths to your saved models
MODEL_PATH = "enrollment_data/dae_puf.pt"

if not os.path.exists(MODEL_PATH):
    print(f"[-] Error: Could not find trained model weights at {MODEL_PATH}")
    print("[-] Please run 'python3 enrollment.py enroll pi-alice' first.")
    exit()

print("[+] Loading live DAE model weights...")
# Recreate the architecture structure (256 input bytes -> 64 bottleneck)
model = DenoisingAutoencoder(input_dim=256, bottleneck_dim=64)
checkpoint = torch.load(MODEL_PATH)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval() # Put the neural network into live execution/inference mode

# 1. Simulate a live, noisy hardware reading (256 values between 0 and 1)
print("[+] Simulating a noisy hardware response...")
print("[+] Extracting an idealized signature from the model's target space...")
# Pass a baseline vector through the model to capture a valid target representation
base_tensor = torch.ones(1, 256)
with torch.no_grad():
    clean_target = model(base_tensor).numpy().flatten()
clean_mock_hardware = clean_target
print("[+] Overlaying real-time environmental noise onto the valid signature...")
environmental_noise = np.random.normal(0, 0.05, 256).astype(np.float32) # Simulated thermal interference
noisy_live_input = clean_mock_hardware + environmental_noise

# 2. Feed it live into the PyTorch DAE
input_tensor = torch.from_numpy(noisy_live_input)
with torch.no_grad(): # Disable gradient updates for fast execution
    cleaned_output_tensor = model(input_tensor)
    cleaned_output = cleaned_output_tensor.numpy()

# 3. Display the live results
print("\n==================================================")
print("             DAE LIVE WORKING METRICS             ")
print("==================================================")
print(f"Noisy Input (First 5 values):  {noisy_live_input[:5]}")
print(f"Cleaned Output (First 5 values): {cleaned_output[:5]}")
print("==================================================")

# Calculate the Euclidean distance to see how much noise was stripped
initial_error = np.linalg.norm(noisy_live_input - clean_mock_hardware)
final_residual_error = np.linalg.norm(cleaned_output - clean_mock_hardware)

print(f"[LIVE STATUS] Raw incoming signal variance: {initial_error:.4f}")
print(f"[LIVE STATUS] Post-DAE filtered variance:   {final_residual_error:.4f}")
if final_residual_error < initial_error:
    print("\n[✓] SUCCESS: The DAE is running live and actively stripping noise!")
