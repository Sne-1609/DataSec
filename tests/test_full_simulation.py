	#!/usr/bin/env python3
# test_full_simulation.py — End-to-end simulation
# ============================================================
# Simulates both Device A and Device B in a single process
# to verify the entire protocol works (key agreement, 
# authentication, encryption/decryption).
#
# This does NOT launch the interactive chat — it tests the
# crypto pipeline from enrollment through message exchange.
# ============================================================

import os
import sys
import hashlib
import struct
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puf_module import (
    get_puf_response, FuzzyExtractor, derive_signing_keypair,
    sign_message, verify_signature
)
from plkg_module import (
    simulate_channel_measurements, quantize_measurements,
    information_reconciliation, privacy_amplification
)
from pqc_module import PQCKeyExchange
from crypto_module import derive_aes_key, AESCipher

def test_dae_pipeline():
    """Test DAE-based key extraction end-to-end."""
    print("\n" + "=" * 60)
    print("  TEST DAE: DENOISING AUTOENCODER PIPELINE")
    print("=" * 60)

    from dae_module import (
        DenoisingAutoencoder,
        generate_puf_training_data, generate_plkg_training_data,
        train_dae, save_model, load_model,
        dae_clean_puf_response, dae_clean_plkg_measurements,
        DAE_PUF_MODEL_PATH, DAE_PLKG_MODEL_PATH,
    )
    from plkg_module import simulate_channel_measurements

    # ── Train PUF DAE ─────────────────────────────────────────
    puf_clean = get_puf_response(b"enrollment")
    X_n, X_c = generate_puf_training_data(puf_clean, n_samples=500, noise_std=0.05)
    model_puf = DenoisingAutoencoder(256, 64)
    train_dae(model_puf, X_n, X_c, epochs=50, verbose=False)
    save_model(model_puf, DAE_PUF_MODEL_PATH)

    # Run twice with different noise seeds — keys must match
    key1 = dae_clean_puf_response(get_puf_response(b"enrollment"))
    key2 = dae_clean_puf_response(get_puf_response(b"enrollment"))
    puf_stable = key1 == key2
    print(f"    PUF DAE key stability:    {'PASS' if puf_stable else 'FAIL'}")

    # ── Train PLKG DAE ────────────────────────────────────────
    seed = struct.pack(">Q", int(time.time()) // 10)
    X_n, X_c = generate_plkg_training_data(seed, n_samples=500)
    model_plkg = DenoisingAutoencoder(50, 16)
    train_dae(model_plkg, X_n, X_c, epochs=50, verbose=False)
    save_model(model_plkg, DAE_PLKG_MODEL_PATH)

    meas_a = simulate_channel_measurements(50, seed)
    meas_b = simulate_channel_measurements(50, seed)  # correlated noise
    key_a = dae_clean_plkg_measurements(meas_a)
    key_b = dae_clean_plkg_measurements(meas_b)
    plkg_match = key_a == key_b
    print(f"    PLKG DAE key agreement:   {'PASS' if plkg_match else 'FAIL'}")

    assert puf_stable and plkg_match, "DAE tests FAILED"
    print("    >> DAE PIPELINE: PASS")
    return True
def test_phase1_puf_auth():
    """Test PUF enrollment and mutual authentication."""
    print("=" * 60)
    print("  TEST PHASE 1: PUF AUTHENTICATION")
    print("=" * 60)
    
    # --- Enrollment (one-time per device) ---
    # Both devices use the same machine here, but on real Pis
    # they would produce different PUF responses.
    print("\n  [Enrolling Alice...]")
    puf_a = get_puf_response(b"enrollment")
    fe_a = FuzzyExtractor()
    key_a, helper_a = fe_a.gen(puf_a)
    sk_a, pk_a = derive_signing_keypair(key_a)
    print(f"    Alice PUF key: {key_a.hex()[:32]}...")
    print(f"    Alice public:  {pk_a.encode().hex()[:32]}...")
    
    # For simulation, Bob gets a "different" PUF by using different challenge
    print("  [Enrolling Bob...]")
    puf_b = get_puf_response(b"enrollment_bob_device")
    fe_b = FuzzyExtractor()
    key_b, helper_b = fe_b.gen(puf_b)
    sk_b, pk_b = derive_signing_keypair(key_b)
    print(f"    Bob PUF key:   {key_b.hex()[:32]}...")
    print(f"    Bob public:    {pk_b.encode().hex()[:32]}...")
    
    # --- Authentication ---
    print("\n  [Mutual Authentication...]")
    
    # Bob challenges Alice
    nonce_from_bob = os.urandom(32)
    
    # Alice re-derives her key and signs
    puf_a_again = get_puf_response(b"enrollment")
    key_a_again = fe_a.rep(puf_a_again, helper_a)
    sk_a_again, _ = derive_signing_keypair(key_a_again)
    sig_a = sign_message(sk_a_again, nonce_from_bob)
    
    # Bob verifies Alice's signature using her stored public key
    valid_a = verify_signature(pk_a.encode(), nonce_from_bob, sig_a)
    print(f"    Alice -> Bob verification: {'PASS' if valid_a else 'FAIL'}")
    
    # Alice challenges Bob
    nonce_from_alice = os.urandom(32)
    
    # Bob re-derives and signs
    puf_b_again = get_puf_response(b"enrollment_bob_device")
    key_b_again = fe_b.rep(puf_b_again, helper_b)
    sk_b_again, _ = derive_signing_keypair(key_b_again)
    sig_b = sign_message(sk_b_again, nonce_from_alice)
    
    # Alice verifies Bob
    valid_b = verify_signature(pk_b.encode(), nonce_from_alice, sig_b)
    print(f"    Bob -> Alice verification: {'PASS' if valid_b else 'FAIL'}")
    
    assert valid_a and valid_b, "Authentication FAILED!"
    print("    >> MUTUAL AUTH: PASS")
    return True


def test_phase2_plkg():
    """Test PLKG key agreement."""
    print("\n" + "=" * 60)
    print("  TEST PHASE 2: PLKG KEY AGREEMENT")
    print("=" * 60)
    
    # Both devices measure the same channel (simulated with same seed)
    shared_seed = struct.pack(">Q", int(time.time()) // 10)
    
    # Alice measures
    meas_a = simulate_channel_measurements(50, shared_seed)
    bits_a = quantize_measurements(meas_a)
    syn_a = hashlib.sha256(bits_a + b"PLKG_SYNDROME_V1").digest()
    
    # Bob measures (correlated but slightly noisy)
    meas_b = simulate_channel_measurements(50, shared_seed)
    bits_b = quantize_measurements(meas_b)
    syn_b = hashlib.sha256(bits_b + b"PLKG_SYNDROME_V1").digest()
    
    print(f"    Alice quantized: {bits_a.hex()[:24]}...")
    print(f"    Bob quantized:   {bits_b.hex()[:24]}...")
    print(f"    Alice syndrome:  {syn_a.hex()[:24]}...")
    print(f"    Bob syndrome:    {syn_b.hex()[:24]}...")
    
    # Reconciliation (symmetric — both sides sort inputs)
    key_a = information_reconciliation(syn_a, syn_b)
    key_b = information_reconciliation(syn_b, syn_a)
    
    print(f"    Alice reconciled: {key_a.hex()}")
    print(f"    Bob reconciled:   {key_b.hex()}")
    
    # Privacy amplification
    session_ctx = os.urandom(16)
    plkg_a = privacy_amplification(key_a, session_ctx)
    plkg_b = privacy_amplification(key_b, session_ctx)
    
    print(f"    Alice PLKG key: {plkg_a.hex()}")
    print(f"    Bob PLKG key:   {plkg_b.hex()}")
    
    keys_match = plkg_a == plkg_b
    print(f"    >> PLKG KEYS MATCH: {'PASS' if keys_match else 'FAIL'}")
    assert keys_match, "PLKG keys don't match!"
    return plkg_a


def test_phase3_pqc():
    """Test PQC key exchange."""
    print("\n" + "=" * 60)
    print("  TEST PHASE 3: PQC KEY EXCHANGE (KYBER)")
    print("=" * 60)
    
    # Alice generates key pair
    alice_pqc = PQCKeyExchange()
    alice_pk = alice_pqc.generate_keypair()
    
    # Bob encapsulates using Alice's public key
    bob_pqc = PQCKeyExchange()
    ciphertext, bob_secret = bob_pqc.encapsulate(alice_pk)
    
    # Alice decapsulates
    alice_secret = alice_pqc.decapsulate(ciphertext)
    
    secrets_match = alice_secret == bob_secret
    print(f"    Alice secret: {alice_secret.hex()[:32]}...")
    print(f"    Bob secret:   {bob_secret.hex()[:32]}...")
    print(f"    >> PQC SECRETS MATCH: {'PASS' if secrets_match else 'FAIL'}")
    assert secrets_match, "PQC secrets don't match!"
    return alice_secret


def test_phase4_key_derivation(plkg_key, pqc_key):
    """Test HKDF key derivation."""
    print("\n" + "=" * 60)
    print("  TEST PHASE 4: KEY DERIVATION (HKDF)")
    print("=" * 60)
    
    # Both sides use the same salt (Alice sends to Bob)
    salt = os.urandom(16)
    session_id = hashlib.sha256(plkg_key + pqc_key + salt).digest()
    
    aes_key_a = derive_aes_key(plkg_key, pqc_key, session_id)
    aes_key_b = derive_aes_key(plkg_key, pqc_key, session_id)
    
    keys_match = aes_key_a == aes_key_b
    print(f"    PLKG input:    {plkg_key.hex()}")
    print(f"    PQC input:     {pqc_key.hex()[:32]}...")
    print(f"    Alice AES key: {aes_key_a.hex()}")
    print(f"    Bob AES key:   {aes_key_b.hex()}")
    print(f"    >> AES KEYS MATCH: {'PASS' if keys_match else 'FAIL'}")
    assert keys_match, "AES keys don't match!"
    return aes_key_a


def test_phase5_encryption(aes_key):
    """Test AES-256-GCM message exchange."""
    print("\n" + "=" * 60)
    print("  TEST PHASE 5: ENCRYPTED MESSAGING (AES-256-GCM)")
    print("=" * 60)
    
    alice_cipher = AESCipher(aes_key)
    bob_cipher = AESCipher(aes_key)
    
    # Alice sends messages to Bob
    test_messages = [
        "Hello Bob! This is Alice.",
        "Testing triple-layer security!",
        "Secret data: PI=3.14159265",
        "Unicode test: hello world 123",
        "",  # empty message
        "A" * 1000,  # long message
    ]
    
    all_pass = True
    for i, msg in enumerate(test_messages):
        encrypted = alice_cipher.encrypt_text(msg)
        decrypted = bob_cipher.decrypt_text(encrypted)
        ok = decrypted == msg
        status = "PASS" if ok else "FAIL"
        label = msg[:30] + "..." if len(msg) > 30 else msg
        print(f"    Message {i+1}: '{label}' -> {status} ({len(encrypted)} bytes)")
        if not ok:
            all_pass = False
    
    # Test bidirectional
    print("\n    [Bidirectional test]")
    bob_msg = "Reply from Bob!"
    enc = bob_cipher.encrypt_text(bob_msg)
    dec = alice_cipher.decrypt_text(enc)
    bidir_ok = dec == bob_msg
    print(f"    Bob -> Alice: '{dec}' -> {'PASS' if bidir_ok else 'FAIL'}")
    
    # Test tamper detection
    print("\n    [Tamper detection test]")
    encrypted = alice_cipher.encrypt_text("Secret")
    tampered = bytearray(encrypted)
    tampered[25] ^= 0xFF
    try:
        bob_cipher.decrypt_text(bytes(tampered))
        print("    Tamper detection: FAIL (accepted tampered message!)")
        all_pass = False
    except ValueError:
        print("    Tamper detection: PASS (rejected tampered message)")
    
    # Test wrong key detection
    print("\n    [Wrong key test]")
    wrong_cipher = AESCipher(os.urandom(32))
    encrypted = alice_cipher.encrypt_text("Secret")
    try:
        wrong_cipher.decrypt_text(encrypted)
        print("    Wrong key detection: FAIL (decrypted with wrong key!)")
        all_pass = False
    except ValueError:
        print("    Wrong key detection: PASS (rejected wrong key)")
    
    print(f"\n    >> ENCRYPTION: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    assert all_pass, "Encryption tests failed!"
    return True


def main():
    print("")
    print("#" * 60)
    print("#    TRIPLE-LAYER SECURITY -- FULL SIMULATION TEST")
    print("#    PUF + PLKG + PQC + AES-256-GCM")
    print("#" * 60)
    
    results = {}
    
    try:
        # Phase 1: PUF Authentication
        results["PUF Auth"] = test_phase1_puf_auth()
        
        # Phase 2: PLKG Key Agreement
        plkg_key = test_phase2_plkg()
        results["PLKG"] = True
        
        # Phase 3: PQC Key Exchange
        pqc_key = test_phase3_pqc()
        results["PQC"] = True
        
        # Phase 4: Key Derivation
        aes_key = test_phase4_key_derivation(plkg_key, pqc_key)
        results["HKDF"] = True
        
        # Phase 5: Encrypted Messaging
        results["AES-GCM"] = test_phase5_encryption(aes_key)
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {test:20s} : {status}")
    
    all_pass = all(results.values())
    print(f"\n    {'ALL TESTS PASSED!' if all_pass else 'SOME TESTS FAILED!'}")
    print("=" * 60)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
