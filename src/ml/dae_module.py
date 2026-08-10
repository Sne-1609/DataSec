# dae_module.py — Denoising Autoencoder for PUF + PLKG signal cleaning
# ============================================================
# Provides two DAE instances:
#   DAE_PUF  — cleans 256-byte PUF responses → stable 32-byte key
#   DAE_PLKG — cleans 50-sample RSSI vectors → stable 128-bit key
#
# Training happens once during enrollment.
# Inference replaces FuzzyExtractor.rep() and quantize_measurements().
# ============================================================

import os
import struct
import hashlib
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    BACKEND = "torch"
except ImportError:
    BACKEND = None

# ── Model file paths ──────────────────────────────────────────
DAE_PUF_MODEL_PATH  = "enrollment_data/dae_puf.pt"
DAE_PLKG_MODEL_PATH = "enrollment_data/dae_plkg.pt"


# ============================================================
# Neural Network Architecture
# ============================================================

class DenoisingAutoencoder(nn.Module):
    """
    Lightweight fully-connected autoencoder.

    Encoder compresses the noisy input to a bottleneck.
    Decoder reconstructs the clean signal from the bottleneck.

    Input/output dimensions are matched to:
      - PUF:  256 bytes → 32-byte key (input_dim=256, bottleneck=64)
      - PLKG: 50 RSSI samples → 16-byte key (input_dim=50, bottleneck=16)
    """

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        hidden = max(input_dim // 2, bottleneck_dim * 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, bottleneck_dim),
            nn.Sigmoid(),   # Output in [0, 1] — matches normalized bits
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        """Return only the bottleneck (used to extract the key)."""
        return self.encoder(x)


# ============================================================
# Training Data Generation
# ============================================================

def generate_puf_training_data(clean_response: bytes,
                                n_samples: int = 2000,
                                noise_std: float = 0.05):
    """
    Generate (noisy_input, clean_target) pairs for PUF training.

    clean_response : The enrollment PUF response (256 bytes).
    n_samples      : How many noisy variants to synthesize.
    noise_std      : Gaussian noise strength (0.05 = ±5% bit flip rate).

    Returns: (X_noisy, X_clean) as float32 numpy arrays, shape (n, 256).
    """
    # Normalize clean response to [0, 1]
    clean_vec = np.frombuffer(clean_response, dtype=np.uint8).astype(np.float32) / 255.0
    X_clean = np.tile(clean_vec, (n_samples, 1))

    # Add Gaussian noise (simulates SRAM/DRAM bit instability)
    noise = np.random.normal(0, noise_std, X_clean.shape).astype(np.float32)
    X_noisy = np.clip(X_clean + noise, 0.0, 1.0)

    return X_noisy, X_clean


def generate_plkg_training_data(shared_seed: bytes,
                                 n_samples: int = 2000,
                                 noise_amplitude: int = 3):
    """
    Generate (noisy_input, clean_target) pairs for PLKG training.

    shared_seed     : Seed used by simulate_channel_measurements() during enrollment.
    n_samples       : Number of noisy measurement sets to synthesize.
    noise_amplitude : Max RSSI noise in dBm (±noise_amplitude).

    Returns: (X_noisy, X_clean) as float32 numpy arrays, shape (n, 50).
    """
    from plkg_module import simulate_channel_measurements

    # Generate the "clean" channel (deterministic from seed)
    clean_meas = simulate_channel_measurements(50, shared_seed)
    # Normalize RSSI from [-90, -30] range to [0, 1]
    clean_vec = (np.array(clean_meas, dtype=np.float32) + 90) / 60.0
    clean_vec = np.clip(clean_vec, 0.0, 1.0)
    X_clean = np.tile(clean_vec, (n_samples, 1))

    # Simulate independent per-device measurement noise
    noise = np.random.uniform(-noise_amplitude / 60.0,
                               noise_amplitude / 60.0,
                               X_clean.shape).astype(np.float32)
    X_noisy = np.clip(X_clean + noise, 0.0, 1.0)

    return X_noisy, X_clean


# ============================================================
# Training
# ============================================================

def train_dae(model: DenoisingAutoencoder,
              X_noisy: np.ndarray,
              X_clean: np.ndarray,
              epochs: int = 100,
              batch_size: int = 64,
              lr: float = 1e-3,
              verbose: bool = True) -> float:
    """
    Train the DAE to reconstruct X_clean from X_noisy.

    Returns: final training loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_n = torch.tensor(X_noisy)
    X_c = torch.tensor(X_clean)

    n = len(X_n)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xn, xc = X_n[idx], X_c[idx]

            optimizer.zero_grad()
            output = model(xn)
            loss = criterion(output, xc)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        best_loss = min(best_loss, avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

    return best_loss


# ============================================================
# Save / Load
# ============================================================

def save_model(model: DenoisingAutoencoder, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  model.encoder[0].in_features,
        "bottleneck": model.encoder[3].out_features,
    }, path)
    print(f"  DAE saved to: {path}")


def load_model(path: str) -> DenoisingAutoencoder:
    ckpt = torch.load(path, map_location="cpu")
    model = DenoisingAutoencoder(ckpt["input_dim"], ckpt["bottleneck"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ============================================================
# Inference — Drop-in replacements
# ============================================================

def dae_clean_puf_response(noisy_response: bytes,
                            model_path: str = DAE_PUF_MODEL_PATH) -> bytes:
    """
    DAE replacement for FuzzyExtractor.rep().

    Takes a noisy PUF response (256 bytes) and returns the
    stable 32-byte key (bottleneck encoding).
    """
    model = load_model(model_path)

    vec = np.frombuffer(noisy_response, dtype=np.uint8).astype(np.float32) / 255.0
    x = torch.tensor(vec).unsqueeze(0)   # shape (1, 256)

    with torch.no_grad():
        bottleneck = model.encode(x)      # shape (1, bottleneck_dim)

    # Convert continuous bottleneck to stable 32-byte key
    raw = bottleneck.squeeze().numpy()
    # Binarize at 0.5 threshold → stable bits
    bits = (raw >= 0.5).astype(np.uint8)
    # Pack bits into bytes
    n_bytes = len(bits) // 8
    key_bytes = bytearray()
    for i in range(n_bytes):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i * 8 + j])
        key_bytes.append(byte)

    # Hash to produce exactly 32 bytes for signing key
    return hashlib.sha256(bytes(key_bytes) + b"DAE_PUF_KEY").digest()


def dae_clean_plkg_measurements(noisy_rssi: list,
                                 model_path: str = DAE_PLKG_MODEL_PATH) -> bytes:
    """
    DAE replacement for quantize_measurements().

    Takes a list of noisy RSSI values and returns a stable
    128-bit (16-byte) PLKG key.
    """
    model = load_model(model_path)

    arr = np.array(noisy_rssi, dtype=np.float32)
    # Normalize from RSSI range [-90, -30] to [0, 1]
    arr = (arr + 90.0) / 60.0
    arr = np.clip(arr, 0.0, 1.0)

    # Pad or truncate to model's expected input_dim
    expected = model.encoder[0].in_features
    if len(arr) < expected:
        arr = np.pad(arr, (0, expected - len(arr)))
    else:
        arr = arr[:expected]

    x = torch.tensor(arr).unsqueeze(0)

    with torch.no_grad():
        bottleneck = model.encode(x)

    raw = bottleneck.squeeze().numpy()
    bits = (raw >= 0.5).astype(np.uint8)

    n_bytes = len(bits) // 8
    key_bytes = bytearray()
    for i in range(n_bytes):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i * 8 + j])
        key_bytes.append(byte)

    return hashlib.sha256(bytes(key_bytes) + b"DAE_PLKG_KEY").digest()[:16]
