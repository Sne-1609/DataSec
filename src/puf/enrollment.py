# enrollment.py — One-time device enrollment script
# ============================================================
# Run this ONCE on each device during initial setup.
# It generates PUF identity, creates signing keys, and exports
# the public key for the peer device.
# ============================================================

import os
import sys
import json

from puf_module import (
    get_puf_response, FuzzyExtractor, derive_signing_keypair,
    save_helper_data
)
from config import ENROLLMENT_DIR, HELPER_DATA_FILE, DEVICE_ID_FILE

from dae_module import (
    DenoisingAutoencoder,
    generate_puf_training_data,
    generate_plkg_training_data,
    train_dae,
    save_model,
    DAE_PUF_MODEL_PATH,
    DAE_PLKG_MODEL_PATH,
)
import struct, time


def train_and_save_dae(puf_response: bytes):
    """
    Train both DAEs using the enrollment PUF response as the
    ground truth "clean" signal. Call this immediately after
    standard enrollment so helper_data is already saved.
    """
    print("\n[DAE] Training Denoising Autoencoders...")

    # ── DAE for PUF ──────────────────────────────────────────
    print("\n  [DAE-PUF] Generating training data...")
    X_noisy_puf, X_clean_puf = generate_puf_training_data(
        puf_response, n_samples=2000, noise_std=0.05
    )

    model_puf = DenoisingAutoencoder(input_dim=256, bottleneck_dim=64)
    print("  [DAE-PUF] Training (100 epochs)...")
    loss = train_dae(model_puf, X_noisy_puf, X_clean_puf, epochs=100)
    save_model(model_puf, DAE_PUF_MODEL_PATH)
    print(f"  [DAE-PUF] Done. Final loss: {loss:.6f}")

    # ── DAE for PLKG ─────────────────────────────────────────
    print("\n  [DAE-PLKG] Generating training data...")
    shared_seed = struct.pack(">Q", int(time.time()) // 10)
    X_noisy_plkg, X_clean_plkg = generate_plkg_training_data(
        shared_seed, n_samples=2000, noise_amplitude=3
    )

    model_plkg = DenoisingAutoencoder(input_dim=50, bottleneck_dim=16)
    print("  [DAE-PLKG] Training (100 epochs)...")
    loss = train_dae(model_plkg, X_noisy_plkg, X_clean_plkg, epochs=100)
    save_model(model_plkg, DAE_PLKG_MODEL_PATH)
    print(f"  [DAE-PLKG] Done. Final loss: {loss:.6f}")

    print("\n  [DAE] Both models trained and saved.")
    print("  Transfer these to the peer device:")
    print(f"    {DAE_PUF_MODEL_PATH}")
    print(f"    {DAE_PLKG_MODEL_PATH}")

def enroll_device(device_id: str) -> bytes:
    """
    Enroll this device:
    3. Derive signing key pair
    4. Save helper data locally
    5. Export public key (transfer to peer)
    """
    print("=" * 60)
    print(f"  DEVICE ENROLLMENT -- {device_id}")
    print("=" * 60)
    
    # Create enrollment directory
    os.makedirs(ENROLLMENT_DIR, exist_ok=True)
    
    # Step 1: Read PUF response
    print("\n[1/5] Reading PUF response...")
    puf_response = get_puf_response(b"enrollment")
    print(f"  PUF response: {puf_response[:16].hex()}... ({len(puf_response)} bytes)")
    
    # Step 2: Generate stable key via fuzzy extractor
    print("\n[2/5] Running fuzzy extractor...")
    fe = FuzzyExtractor()
    puf_key, helper_data = fe.gen(puf_response)
    print(f"  Stable PUF key: {puf_key.hex()[:32]}...")
    
    # Step 3: Derive signing key pair
    print("\n[3/5] Deriving signing key pair...")
    sk, pk = derive_signing_keypair(puf_key)
    public_key_bytes = pk.encode()
    print(f"  Public key: {public_key_bytes.hex()[:32]}...")
    
    # Step 4: Save helper data and device ID locally
    print("\n[4/5] Saving enrollment data...")
    save_helper_data(helper_data, HELPER_DATA_FILE)
    
    with open(DEVICE_ID_FILE, "w") as f:
        f.write(device_id)
    print(f"  Device ID saved to: {DEVICE_ID_FILE}")
    
    # Step 5: Export public key to file (for transfer to peer)
    pubkey_export = os.path.join(ENROLLMENT_DIR, f"{device_id}_public_key.bin")
    with open(pubkey_export, "wb") as f:
        f.write(public_key_bytes)
    print(f"  Public key exported to: {pubkey_export}")
    
    # Also export as hex text for easy copy
    pubkey_hex_file = os.path.join(ENROLLMENT_DIR, f"{device_id}_public_key.hex")
    with open(pubkey_hex_file, "w") as f:
        f.write(public_key_bytes.hex())
    print(f"  Public key (hex) exported to: {pubkey_hex_file}")
    
    print("\n" + "=" * 60)
    print("  [OK] ENROLLMENT COMPLETE")
    print("=" * 60)
    print(f"""
  NEXT STEPS:
  1. Copy '{pubkey_export}' to the OTHER Raspberry Pi
  2. On the other Pi, save it as:
     {ENROLLMENT_DIR}/peer_public_key.bin
  3. Do the same enrollment on the other Pi
  4. Exchange public keys between both devices
  
  Example (using SCP from this Pi to the other):
    scp {pubkey_export} pi@<OTHER_PI_IP>:~/triple_layer_security/{ENROLLMENT_DIR}/peer_public_key.bin
""")
    
    #return puf_respose
    return public_key_bytes
    return puf_response


def import_peer_key(peer_key_file: str):
    """Import a peer's public key."""
    from config import PEER_PUBKEY_FILE
    
    if peer_key_file.endswith(".hex"):
        with open(peer_key_file, "r") as f:
            key_bytes = bytes.fromhex(f.read().strip())
    else:
        with open(peer_key_file, "rb") as f:
            key_bytes = f.read()
    
    os.makedirs(ENROLLMENT_DIR, exist_ok=True)
    with open(PEER_PUBKEY_FILE, "wb") as f:
        f.write(key_bytes)
    
    print(f"[OK] Peer public key imported: {PEER_PUBKEY_FILE}")
    print(f"   Key: {key_bytes.hex()[:32]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Enroll this device:")
        print("    python3 enrollment.py enroll <device_id>")
        print("    Example: python3 enrollment.py enroll pi-alice")
        print()
        print("  Import peer's public key:")
        print("    python3 enrollment.py import <path_to_peer_key>")
        print("    Example: python3 enrollment.py import pi-bob_public_key.bin")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "enroll":
        device_id = sys.argv[2] if len(sys.argv) > 2 else "device-001"
        puf_bytes = enroll_device(device_id)
	if "--dae" in sys.argv:
            train_and_save_dae(puf_bytes)
    #enroll_device(device_id)    
    elif command == "import":
        if len(sys.argv) < 3:
            print("Error: specify path to peer's public key file")
            sys.exit(1)
        import_peer_key(sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
