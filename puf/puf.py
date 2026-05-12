# puf_module.py — PUF Simulation + Fuzzy Extractor
# ============================================================
# On Raspberry Pi, we simulate a PUF using device-specific
# hardware identifiers (CPU serial, MAC address) combined with
# entropy. For a real PUF, replace get_puf_response() with
# actual DRAM decay measurement or external PUF chip reading.
# ============================================================

import hashlib
import os
import json
import struct


def get_device_serial() -> str:
    """Read Raspberry Pi CPU serial number (unique per device)."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.strip().split(":")[1].strip()
    except:
        pass
    # Fallback: use hostname + MAC as identifier
    import uuid
    return str(uuid.getnode())


def get_puf_response(challenge: bytes = b"default") -> bytes:
    """
    Simulate a PUF response using device-specific hardware properties.
    
    In a real implementation, replace this with:
    - DRAM decay PUF: Disable DRAM refresh, read decay patterns
    - SRAM PUF: Read uninitialized SRAM power-on state  
    - External PUF chip: Query via I2C/SPI (e.g., ATECC608B)
    
    For this demo: We derive a device-unique, reproducible response
    from the CPU serial number + challenge. The response is
    DETERMINISTIC per device — same device always gives the same output.
    """
    serial = get_device_serial()
    
    # Deterministic base response from hardware identity
    base = hashlib.sha512(
        serial.encode() + challenge + b"PUF_SEED_v1"
    ).digest()
    
    # Extend to desired length (256 bytes = 2048 bits)
    response = b""
    for i in range(4):
        response += hashlib.sha512(
            base + struct.pack(">I", i)
        ).digest()
    
    return response[:256]  # 256 bytes = 2048 bits


class FuzzyExtractor:
    """
    Converts PUF responses into stable cryptographic keys.
    
    Since our simulated PUF is deterministic (same device = same output),
    the fuzzy extractor uses a straightforward key derivation approach.
    
    For a REAL noisy PUF, replace this with BCH/Reed-Solomon error
    correction codes (e.g., using the 'galois' Python package).
    """
    
    def __init__(self, key_length: int = 32):
        self.key_length = key_length
    
    def gen(self, puf_response: bytes) -> tuple:
        """
        ENROLLMENT — Run once in a secure environment.
        
        Returns: (key, helper_data)
        - key: stable 256-bit key derived from PUF
        - helper_data: public data for key reconstruction
        """
        # Derive a stable key from the PUF response
        key = hashlib.sha256(
            puf_response + b"STABLE_KEY_DERIVATION_V2"
        ).digest()[:self.key_length]
        
        # Store a commitment (hash of key) for verification
        helper_data = {
            "commitment": hashlib.sha256(key).hexdigest(),
            "response_hash": hashlib.sha256(puf_response).hexdigest(),
        }
        
        return key, helper_data
    
    def rep(self, puf_response: bytes, helper_data: dict) -> bytes:
        """
        REPRODUCTION — Run at each authentication.
        Recovers the same key from the PUF response + helper data.
        
        Returns: the same key as gen() if device is genuine.
        Raises ValueError if PUF response doesn't match.
        """
        # Re-derive key
        key = hashlib.sha256(
            puf_response + b"STABLE_KEY_DERIVATION_V2"
        ).digest()[:self.key_length]
        
        # Verify against stored commitment
        commitment = hashlib.sha256(key).hexdigest()
        
        if commitment == helper_data["commitment"]:
            return key
        
        raise ValueError(
            "PUF response mismatch — key recovery failed. "
            "Device may not be genuine or PUF has drifted."
        )


def save_helper_data(helper_data: dict, filepath: str):
    """Save helper data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(helper_data, f, indent=2)
    print(f"  Helper data saved to: {filepath}")


def load_helper_data(filepath: str) -> dict:
    """Load helper data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


# ============================================================
# PUF AUTHENTICATION (Public-Key Signature based)
# ============================================================

def derive_signing_keypair(puf_seed: bytes):
    """
    Derive an Ed25519 signing key pair from PUF seed.
    The private key is re-derived each time from PUF — never stored.
    """
    from nacl.signing import SigningKey
    seed_32 = hashlib.sha256(puf_seed + b"SIGNING_KEY").digest()
    sk = SigningKey(seed_32)
    pk = sk.verify_key
    return sk, pk


def sign_message(private_key, message: bytes) -> bytes:
    """Sign a message with PUF-derived private key."""
    signed = private_key.sign(message)
    return signed.signature


def verify_signature(public_key_bytes: bytes, message: bytes, 
                     signature: bytes) -> bool:
    """Verify a signature using the stored public key."""
    from nacl.signing import VerifyKey
    try:
        vk = VerifyKey(public_key_bytes)
        vk.verify(message, signature)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print("=== PUF Module Test ===")
    
    # Get PUF response (deterministic per device)
    response = get_puf_response()
    print(f"PUF response: {response[:16].hex()}... ({len(response)} bytes)")
    
    # Fuzzy extractor enrollment
    fe = FuzzyExtractor()
    key, helper = fe.gen(response)
    print(f"Derived key: {key.hex()}")
    
    # Test recovery (same device → same response → same key)
    response2 = get_puf_response()
    try:
        key2 = fe.rep(response2, helper)
        print(f"Recovered key: {key2.hex()}")
        print(f"Keys match: {key == key2}")
    except ValueError as e:
        print(f"Recovery failed: {e}")
    
    # Test signing
    sk, pk = derive_signing_keypair(key)
    nonce = os.urandom(32)
    sig = sign_message(sk, nonce)
    valid = verify_signature(pk.encode(), nonce, sig)
    print(f"Signature valid: {valid}")
