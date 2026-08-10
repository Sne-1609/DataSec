# pqc_module.py — Post-Quantum Cryptography (Kyber/ML-KEM)
# ============================================================
# Handles NIST FIPS 203 ML-KEM-768 (NIST Level 3, ~AES-192)
# key encapsulation mechanism for quantum-resistant key
# exchange between the two devices.
#
# Backend priority:
#   1. kyber-py  (pure Python ML-KEM-768 — works everywhere)
#   2. liboqs    (C library via oqs — fastest, needs native lib)
#   3. Simulated (X25519 fallback — NOT quantum-resistant)
# ============================================================

import time

ALGORITHM = "ML-KEM-768"  # NIST Level 3, ~AES-192 security


# ── Backend detection ────────────────────────────────────────

def _check_kyber_py():
    """Check if kyber-py is installed."""
    try:
        from kyber_py.ml_kem import ML_KEM_768
        return True
    except ImportError:
        return False


def _check_liboqs():
    """Check if liboqs is installed and Kyber is available."""
    try:
        import oqs
        mechs = oqs.get_enabled_kem_mechanisms()
        for name in ["ML-KEM-768", "Kyber768"]:
            if name in mechs:
                return True, name
        return False, None
    except (ImportError, RuntimeError):
        return False, None


def check_pqc_available():
    """
    Check which PQC backend is available.
    Returns: (backend_name, liboqs_algo_name_or_None)
    """
    if _check_kyber_py():
        return "kyber-py", None

    available, algo = _check_liboqs()
    if available:
        return "liboqs", algo

    return None, None


class PQCKeyExchange:
    """
    Post-Quantum Key Exchange using CRYSTALS-Kyber (ML-KEM-768).
    
    NIST Level 3 — equivalent to AES-192 security.
    
    Flow:
    1. Alice: generate_keypair() -> (public_key, stored secret_key)
    2. Bob:   encapsulate(alice_public_key) -> (ciphertext, shared_secret)
    3. Alice: decapsulate(ciphertext) -> shared_secret
    
    Both now have the same 256-bit shared secret.
    """
    
    def __init__(self):
        backend, liboqs_algo = check_pqc_available()
        
        if backend == "kyber-py":
            self.backend = "kyber-py"
            self.simulated = False
            print(f"  Using PQC: {ALGORITHM} (NIST Level 3) via kyber-py")
        elif backend == "liboqs":
            self.backend = "liboqs"
            self._liboqs_algo = liboqs_algo
            self.simulated = False
            print(f"  Using PQC: {ALGORITHM} (NIST Level 3) via liboqs")
        else:
            print("  [!] No real PQC library available -- using simulated PQC")
            self.backend = "simulated"
            self.simulated = True
        
        self.algorithm = ALGORITHM
        self.public_key = None
        self.secret_key = None
        self.shared_secret = None
    
    def generate_keypair(self) -> bytes:
        """
        Generate a Kyber key pair.
        Returns the public key (send to peer).
        Secret key is stored internally.
        """
        if self.backend == "kyber-py":
            return self._kyberpy_generate_keypair()
        elif self.backend == "liboqs":
            return self._liboqs_generate_keypair()
        else:
            return self._sim_generate_keypair()
    
    def encapsulate(self, peer_public_key: bytes) -> tuple:
        """
        Encapsulate: create ciphertext + shared secret using peer's public key.
        Returns: (ciphertext, shared_secret)
        """
        if self.backend == "kyber-py":
            return self._kyberpy_encapsulate(peer_public_key)
        elif self.backend == "liboqs":
            return self._liboqs_encapsulate(peer_public_key)
        else:
            return self._sim_encapsulate(peer_public_key)
    
    def decapsulate(self, ciphertext: bytes) -> bytes:
        """
        Decapsulate: recover shared secret from ciphertext using secret key.
        Returns: shared_secret (same as encapsulator's)
        """
        if self.backend == "kyber-py":
            return self._kyberpy_decapsulate(ciphertext)
        elif self.backend == "liboqs":
            return self._liboqs_decapsulate(ciphertext)
        else:
            return self._sim_decapsulate(ciphertext)

    # ── kyber-py backend (real ML-KEM-768) ───────────────────

    def _kyberpy_generate_keypair(self) -> bytes:
        """Generate ML-KEM-768 key pair using kyber-py."""
        from kyber_py.ml_kem import ML_KEM_768

        start = time.perf_counter_ns()
        self.public_key, self.secret_key = ML_KEM_768.keygen()
        elapsed_us = (time.perf_counter_ns() - start) / 1000

        print(f"  KeyGen complete ({elapsed_us:.0f} us)")
        print(f"  Public key:  {len(self.public_key)} bytes")
        print(f"  Secret key:  {len(self.secret_key)} bytes")

        return self.public_key

    def _kyberpy_encapsulate(self, peer_public_key: bytes) -> tuple:
        """Encapsulate using kyber-py ML-KEM-768."""
        from kyber_py.ml_kem import ML_KEM_768

        start = time.perf_counter_ns()
        self.shared_secret, ciphertext = ML_KEM_768.encaps(peer_public_key)
        elapsed_us = (time.perf_counter_ns() - start) / 1000

        print(f"  Encapsulation complete ({elapsed_us:.0f} us)")
        print(f"  Ciphertext:  {len(ciphertext)} bytes")
        print(f"  Shared secret: {self.shared_secret.hex()[:32]}...")

        return ciphertext, self.shared_secret

    def _kyberpy_decapsulate(self, ciphertext: bytes) -> bytes:
        """Decapsulate using kyber-py ML-KEM-768."""
        from kyber_py.ml_kem import ML_KEM_768

        start = time.perf_counter_ns()
        self.shared_secret = ML_KEM_768.decaps(self.secret_key, ciphertext)
        elapsed_us = (time.perf_counter_ns() - start) / 1000

        print(f"  Decapsulation complete ({elapsed_us:.0f} us)")
        print(f"  Shared secret: {self.shared_secret.hex()[:32]}...")

        return self.shared_secret

    # ── liboqs backend (C library) ───────────────────────────

    def _liboqs_generate_keypair(self) -> bytes:
        """Generate Kyber key pair using liboqs."""
        import oqs
        kem = oqs.KeyEncapsulation(self._liboqs_algo)
        
        start = time.perf_counter_ns()
        self.public_key = kem.generate_keypair()
        self.secret_key = kem.export_secret_key()
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        
        print(f"  KeyGen complete ({elapsed_us:.0f} us)")
        print(f"  Public key:  {len(self.public_key)} bytes")
        print(f"  Secret key:  {len(self.secret_key)} bytes")
        
        return self.public_key
    
    def _liboqs_encapsulate(self, peer_public_key: bytes) -> tuple:
        """Encapsulate using liboqs."""
        import oqs
        kem = oqs.KeyEncapsulation(self._liboqs_algo)
        
        start = time.perf_counter_ns()
        ciphertext, self.shared_secret = kem.encap_secret(peer_public_key)
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        
        print(f"  Encapsulation complete ({elapsed_us:.0f} us)")
        print(f"  Ciphertext:  {len(ciphertext)} bytes")
        print(f"  Shared secret: {self.shared_secret.hex()[:32]}...")
        
        return ciphertext, self.shared_secret
    
    def _liboqs_decapsulate(self, ciphertext: bytes) -> bytes:
        """Decapsulate using liboqs."""
        import oqs
        kem = oqs.KeyEncapsulation(self._liboqs_algo, self.secret_key)
        
        start = time.perf_counter_ns()
        self.shared_secret = kem.decap_secret(ciphertext)
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        
        print(f"  Decapsulation complete ({elapsed_us:.0f} us)")
        print(f"  Shared secret: {self.shared_secret.hex()[:32]}...")
        
        return self.shared_secret
    
    # ── Simulated fallback (when no PQC library is installed) ──
    
    def _sim_generate_keypair(self) -> bytes:
        """Simulated key generation using classical crypto."""
        import os
        from nacl.public import PrivateKey
        
        sk = PrivateKey.generate()
        self._sim_sk = sk
        self.public_key = bytes(sk.public_key)
        self.secret_key = bytes(sk)
        
        print(f"  [SIMULATED] KeyGen complete")
        print(f"  Public key:  {len(self.public_key)} bytes")
        
        return self.public_key
    
    def _sim_encapsulate(self, peer_public_key: bytes) -> tuple:
        """Simulated encapsulation using X25519 + random."""
        import os
        import hashlib
        from nacl.public import PrivateKey, PublicKey, Box
        
        # Ephemeral key
        eph_sk = PrivateKey.generate()
        peer_pk = PublicKey(peer_public_key)
        box = Box(eph_sk, peer_pk)
        
        # Shared secret from DH
        self.shared_secret = hashlib.sha256(
            bytes(box.shared_key()) + b"KEM_SHARED_SECRET"
        ).digest()
        
        # "Ciphertext" is the ephemeral public key
        ciphertext = bytes(eph_sk.public_key)
        
        print(f"  [SIMULATED] Encapsulation complete")
        print(f"  Shared secret: {self.shared_secret.hex()[:32]}...")
        
        return ciphertext, self.shared_secret
    
    def _sim_decapsulate(self, ciphertext: bytes) -> bytes:
        """Simulated decapsulation."""
        import hashlib
        from nacl.public import PrivateKey, PublicKey, Box
        
        my_sk = PrivateKey(self.secret_key)
        eph_pk = PublicKey(ciphertext)
        box = Box(my_sk, eph_pk)
        
        self.shared_secret = hashlib.sha256(
            bytes(box.shared_key()) + b"KEM_SHARED_SECRET"
        ).digest()
        
        print(f"  [SIMULATED] Decapsulation complete")
        print(f"  Shared secret: {self.shared_secret.hex()[:32]}...")
        
        return self.shared_secret


if __name__ == "__main__":
    print("=== PQC Module Test (ML-KEM-768) ===")
    
    # Alice generates key pair
    alice = PQCKeyExchange()
    alice_pk = alice.generate_keypair()
    
    print()
    
    # Bob encapsulates
    bob = PQCKeyExchange()
    ct, bob_secret = bob.encapsulate(alice_pk)
    
    print()
    
    # Alice decapsulates
    alice_secret = alice.decapsulate(ct)
    
    print()
    print(f"Secrets match: {alice_secret == bob_secret}")
    print(f"Shared secret: {alice_secret.hex()}")
