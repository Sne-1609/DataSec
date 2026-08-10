# crypto_module.py — AES-256-GCM Encryption + HKDF Key Derivation
# ============================================================
# Handles the final encryption layer:
# - HKDF to combine PLKG + PQC keys into a single AES-256 key
# - AES-256-GCM authenticated encryption for messages
# ============================================================

import hashlib
import hmac
import struct
import os


# ============================================================
# HKDF (HMAC-based Key Derivation Function) — RFC 5869
# ============================================================

def hkdf_extract(salt: bytes, input_key_material: bytes) -> bytes:
    """HKDF-Extract: Extracts a pseudorandom key from input material."""
    if salt is None or len(salt) == 0:
        salt = b"\x00" * 32  # Default salt
    return hmac.new(salt, input_key_material, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-Expand: Expands PRK into output key material."""
    hash_len = 32  # SHA-256
    n = (length + hash_len - 1) // hash_len
    
    okm = b""
    t = b""
    
    for i in range(1, n + 1):
        t = hmac.new(
            prk, t + info + struct.pack("B", i), hashlib.sha256
        ).digest()
        okm += t
    
    return okm[:length]


def derive_aes_key(plkg_key: bytes, pqc_key: bytes, 
                   session_id: bytes = b"") -> bytes:
    """
    Combine PLKG key (128 bits) and PQC key (256 bits) into 
    a single AES-256 key using HKDF.
    
    This is the CORRECT way to combine multiple key sources
    (NOT concatenation!).
    """
    # Combine both key materials
    input_key_material = plkg_key + pqc_key
    
    # Extract phase
    salt = session_id if session_id else os.urandom(32)
    prk = hkdf_extract(salt, input_key_material)
    
    # Expand phase with context
    info = b"triple-layer-security-aes-256-gcm-key-v1"
    aes_key = hkdf_expand(prk, info, length=32)  # 256 bits
    
    return aes_key


# ============================================================
# AES-256-GCM Authenticated Encryption
# ============================================================

class AESCipher:
    """
    AES-256-GCM encryption/decryption.
    
    GCM mode provides:
    - Confidentiality (encryption)
    - Integrity (authentication tag)
    - No padding needed (stream cipher mode)
    """
    
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError(f"AES-256 requires 32-byte key, got {len(key)}")
        self.key = key
        self.message_counter = 0
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt a message with AES-256-GCM.
        
        Returns: nonce (12 bytes) + ciphertext + tag (16 bytes)
        """
        from Crypto.Cipher import AES
        
        # Generate unique nonce for each message
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        
        # Add associated data (message counter for ordering)
        self.message_counter += 1
        aad = struct.pack(">Q", self.message_counter)
        cipher.update(aad)
        
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        
        # Pack: counter (8) + nonce (12) + ciphertext (var) + tag (16)
        return aad + nonce + ciphertext + tag
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt an AES-256-GCM encrypted message.
        
        Input: counter (8) + nonce (12) + ciphertext (var) + tag (16)
        Returns: plaintext bytes
        Raises ValueError if authentication fails (tampered data).
        """
        from Crypto.Cipher import AES
        
        if len(encrypted_data) < 36:  # 8 + 12 + 0 + 16 minimum
            raise ValueError("Encrypted data too short")
        
        # Unpack
        aad = encrypted_data[:8]
        nonce = encrypted_data[8:20]
        ciphertext = encrypted_data[20:-16]
        tag = encrypted_data[-16:]
        
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        cipher.update(aad)
        
        try:
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            return plaintext
        except ValueError:
            raise ValueError(
                "AUTHENTICATION FAILED -- message was tampered with "
                "or wrong key was used!"
            )
    
    def encrypt_text(self, text: str) -> bytes:
        """Convenience: encrypt a string."""
        return self.encrypt(text.encode("utf-8"))
    
    def decrypt_text(self, encrypted_data: bytes) -> str:
        """Convenience: decrypt to string."""
        return self.decrypt(encrypted_data).decode("utf-8")


if __name__ == "__main__":
    print("=== Crypto Module Test ===")
    
    # Simulate key derivation
    plkg_key = os.urandom(16)  # 128 bits from PLKG
    pqc_key = os.urandom(32)   # 256 bits from PQC
    
    print(f"PLKG key: {plkg_key.hex()}")
    print(f"PQC key:  {pqc_key.hex()}")
    
    aes_key = derive_aes_key(plkg_key, pqc_key, b"test-session")
    print(f"AES key:  {aes_key.hex()}")
    
    # Test encryption/decryption
    cipher = AESCipher(aes_key)
    
    messages = [
        "Hello from Device A!",
        "This is a secure message.",
        "Triple-layer encryption working!",
    ]
    
    for msg in messages:
        encrypted = cipher.encrypt_text(msg)
        print(f"\nOriginal:  {msg}")
        print(f"Encrypted: {encrypted.hex()[:40]}... ({len(encrypted)} bytes)")
    
    # Test decryption
    cipher2 = AESCipher(aes_key)  # Same key
    for msg in messages:
        encrypted = cipher.encrypt_text(msg)
        decrypted = cipher2.decrypt_text(encrypted)
        print(f"Decrypted: {decrypted}")
        assert decrypted == msg
    
    print("\nAll encryption/decryption tests passed!")
    
    # Test tamper detection
    print("\nTesting tamper detection...")
    encrypted = cipher.encrypt_text("Secret")
    tampered = bytearray(encrypted)
    tampered[25] ^= 0xFF  # Flip a byte
    try:
        cipher2.decrypt_text(bytes(tampered))
        print("Tampered message was accepted — BAD!")
    except ValueError as e:
        print(f"Tamper detected: {e}")
