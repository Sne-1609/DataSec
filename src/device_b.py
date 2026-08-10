#!/usr/bin/env python3
# device_b.py — Device B (Bob / Client)
# ============================================================
# Run this on Raspberry Pi #2.
# This device acts as the CLIENT — it connects to Device A
# and participates in the security handshake.
# ============================================================

import socket
import os
import sys
import time
import struct
import hashlib
import threading
import json

from config import *
from protocol import send_message, recv_message
from puf_module import (
    get_puf_response, FuzzyExtractor, derive_signing_keypair,
    sign_message, verify_signature, load_helper_data
)
from plkg_module import PLKGKeyGenerator, simulate_channel_measurements, quantize_measurements
from pqc_module import PQCKeyExchange
from crypto_module import derive_aes_key, AESCipher

# ANSI colors
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_banner():
    print(f"""{C.BLUE}{C.BOLD}
╔══════════════════════════════════════════════════════════╗
║          TRIPLE-LAYER SECURITY — DEVICE B (Bob)          ║
║              PUF + PLKG + PQC/AES-256-GCM                ║
╠══════════════════════════════════════════════════════════╣
║  Role:   CLIENT (connecting to Device A)                 ║
║  Layers: 1. PUF Authentication                           ║
║          2. PLKG Key Agreement                           ║
║          3. PQC (Kyber) Key Exchange                     ║
║          4. AES-256-GCM Encrypted Chat                   ║
╚══════════════════════════════════════════════════════════╝
{C.END}""")


def load_enrollment_data():
    """Load PUF helper data and peer's public key."""
    print(f"\n{C.YELLOW}[SETUP] Loading enrollment data...{C.END}")
    
    if not os.path.exists(HELPER_DATA_FILE):
        print(f"{C.RED}ERROR: No enrollment data found!")
        print(f"Run: python3 enrollment.py enroll pi-bob{C.END}")
        sys.exit(1)
    
    helper_data = load_helper_data(HELPER_DATA_FILE)
    print(f"  ✓ Helper data loaded")
    
    with open(DEVICE_ID_FILE, "r") as f:
        device_id = f.read().strip()
    print(f"  ✓ Device ID: {device_id}")
    
    if not os.path.exists(PEER_PUBKEY_FILE):
        print(f"{C.RED}ERROR: Peer public key not found!")
        print(f"Import it: python3 enrollment.py import <peer_key_file>{C.END}")
        sys.exit(1)
    
    with open(PEER_PUBKEY_FILE, "rb") as f:
        peer_pubkey = f.read()
    print(f"  ✓ Peer public key loaded ({len(peer_pubkey)} bytes)")
    
    return helper_data, device_id, peer_pubkey


def phase1_authenticate(sock, helper_data, device_id, peer_pubkey):
    """
    Phase 1: Mutual PUF Authentication
    
    Bob (client) flow:
    1. Send challenge to Alice → receive signature → verify
    2. Receive challenge from Alice → sign with PUF key → send signature
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 1: PUF AUTHENTICATION")
    print(f"{'='*60}{C.END}")
    
    # Re-derive PUF signing key
    print(f"  {C.CYAN}Extracting PUF response...{C.END}")
    puf_response = get_puf_response(b"enrollment")
    fe = FuzzyExtractor()
    puf_key = fe.rep(puf_response, helper_data)
    sk, pk = derive_signing_keypair(puf_key)
    print(f"  {C.GREEN}✓ PUF key recovered, signing key derived{C.END}")
    
    # Step 1: Send challenge to Alice
    our_challenge = os.urandom(NONCE_BYTES)
    send_message(sock, MSG_AUTH_CHALLENGE, our_challenge)
    print(f"  {C.CYAN}Sent challenge to Alice{C.END}")
    
    # Receive Alice's signature
    msg_type, auth_data = recv_message(sock)
    assert msg_type == MSG_AUTH_RESPONSE, f"Expected AUTH_RESPONSE, got {msg_type}"
    peer_id, peer_sig = auth_data.split(b"|", 1)
    
    # Verify Alice's signature
    valid = verify_signature(peer_pubkey, our_challenge, peer_sig)
    if not valid:
        send_message(sock, MSG_AUTH_FAIL)
        print(f"  {C.RED}✗ Alice's signature is INVALID — possible impersonation!{C.END}")
        return False
    print(f"  {C.GREEN}✓ Alice verified: {peer_id.decode()}{C.END}")
    
    # Step 2: Receive Alice's challenge, sign it
    msg_type, challenge_from_alice = recv_message(sock)
    assert msg_type == MSG_AUTH_CHALLENGE
    
    signature = sign_message(sk, challenge_from_alice)
    send_message(sock, MSG_AUTH_RESPONSE, 
                 device_id.encode() + b"|" + signature)
    print(f"  {C.GREEN}✓ Signed Alice's challenge → sent signature{C.END}")
    
    # Wait for Alice's verification result
    msg_type, _ = recv_message(sock)
    if msg_type == MSG_AUTH_OK:
        print(f"  {C.GREEN}{C.BOLD}✓ MUTUAL AUTHENTICATION SUCCESSFUL{C.END}")
        return True
    else:
        print(f"  {C.RED}✗ Alice rejected our authentication{C.END}")
        return False


def phase2_plkg(sock, peer_ip):
    """
    Phase 2: PLKG Key Agreement
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 2: PLKG KEY AGREEMENT")
    print(f"{'='*60}{C.END}")
    
    # Wait for Alice's start signal
    msg_type, _ = recv_message(sock)
    assert msg_type == MSG_PLKG_START
    
    plkg = PLKGKeyGenerator("bob", peer_ip)
    
    # Measure channel  
    plkg.measure_channel(num_samples=50)
    plkg.quantize()
    
    # Signal ready
    send_message(sock, MSG_PLKG_DONE)
    
    # Exchange syndromes
    msg_type, peer_syndrome = recv_message(sock)
    assert msg_type == MSG_PLKG_DATA
    
    our_syndrome = plkg.get_syndrome()
    send_message(sock, MSG_PLKG_DATA, our_syndrome)
    
    # Reconcile and amplify
    plkg.reconcile(peer_syndrome)
    
    msg_type, session_ctx = recv_message(sock)
    assert msg_type == MSG_PLKG_DATA
    plkg_key = plkg.amplify(session_ctx)
    
    print(f"  {C.GREEN}{C.BOLD}✓ PLKG key generated: {plkg_key.hex()}{C.END}")
    return plkg_key


def phase3_pqc(sock):
    """
    Phase 3: PQC Key Exchange (Kyber)
    
    Bob receives Alice's public key, encapsulates, sends ciphertext.
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 3: PQC KEY EXCHANGE (KYBER)")
    print(f"{'='*60}{C.END}")
    
    pqc = PQCKeyExchange()
    
    # Receive Alice's public key
    msg_type, alice_pubkey = recv_message(sock)
    assert msg_type == MSG_PQC_PUBKEY
    print(f"  {C.CYAN}Received Alice's public key ({len(alice_pubkey)} bytes){C.END}")
    
    # Encapsulate
    print(f"  {C.CYAN}Encapsulating shared secret...{C.END}")
    ciphertext, pqc_key = pqc.encapsulate(alice_pubkey)
    
    # Send ciphertext to Alice
    send_message(sock, MSG_PQC_CIPHERTEXT, ciphertext)
    print(f"  {C.GREEN}✓ Ciphertext sent to Alice ({len(ciphertext)} bytes){C.END}")
    
    print(f"  {C.GREEN}{C.BOLD}✓ PQC shared secret: {pqc_key.hex()[:32]}...{C.END}")
    return pqc_key


def phase4_derive_key(sock, plkg_key, pqc_key):
    """
    Phase 4: Derive final AES-256 key from PLKG + PQC via HKDF.
    Bob receives Alice's salt to derive the same session_id.
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 4: KEY DERIVATION (HKDF)")
    print(f"{'='*60}{C.END}")
    
    # Receive salt from Alice to ensure both sides use same session_id
    msg_type, salt = recv_message(sock)
    assert msg_type == MSG_KEY_READY
    
    session_id = hashlib.sha256(
        plkg_key + pqc_key + salt
    ).digest()
    
    aes_key = derive_aes_key(plkg_key, pqc_key, session_id)
    
    print(f"  PLKG key (128-bit): {plkg_key.hex()}")
    print(f"  PQC key  (256-bit): {pqc_key.hex()[:32]}...")
    print(f"  {C.GREEN}{C.BOLD}✓ AES-256 key: {aes_key.hex()}{C.END}")
    
    return aes_key


def phase5_secure_chat(sock, aes_key):
    """
    Phase 5: Real-time encrypted chat using AES-256-GCM.
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 5: SECURE CHAT (AES-256-GCM)")
    print(f"{'='*60}{C.END}")
    print(f"  {C.GREEN}All messages are encrypted with AES-256-GCM{C.END}")
    print(f"  Type your message and press Enter. Type 'quit' to exit.\n")
    
    cipher = AESCipher(aes_key)
    running = True
    
    def receive_messages():
        """Background thread to receive and decrypt messages."""
        nonlocal running
        while running:
            try:
                msg_type, encrypted_payload = recv_message(sock)
                
                if msg_type == MSG_QUIT:
                    print(f"\n  {C.YELLOW}[Alice has left the chat]{C.END}")
                    running = False
                    break
                
                if msg_type == MSG_CHAT:
                    try:
                        plaintext = cipher.decrypt_text(encrypted_payload)
                        print(f"\r  {C.CYAN}{C.BOLD}[Alice]{C.END} {plaintext}")
                        print(f"  {C.GREEN}[You]{C.END} ", end="", flush=True)
                    except ValueError as e:
                        print(f"\n  {C.RED}⚠ TAMPERED MESSAGE DETECTED: {e}{C.END}")
                        
            except ConnectionError:
                print(f"\n  {C.RED}[Connection lost]{C.END}")
                running = False
                break
            except Exception as e:
                if running:
                    print(f"\n  {C.RED}[Error: {e}]{C.END}")
                break
    
    # Synchronize ready state BEFORE starting receiver thread
    send_message(sock, MSG_KEY_READY)
    msg_type, _ = recv_message(sock)  # Wait for Alice's ready
    
    # Start receiver thread AFTER handshake is complete
    recv_thread = threading.Thread(target=receive_messages, daemon=True)
    recv_thread.start()
    
    print(f"  {C.GREEN}{C.BOLD}━━━ SECURE CHANNEL ESTABLISHED ━━━{C.END}\n")
    
    while running:
        try:
            print(f"  {C.GREEN}[You]{C.END} ", end="", flush=True)
            user_input = input()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                send_message(sock, MSG_QUIT)
                running = False
                print(f"  {C.YELLOW}[Chat ended]{C.END}")
                break
            
            # Encrypt and send
            encrypted = cipher.encrypt_text(user_input)
            send_message(sock, MSG_CHAT, encrypted)
            
        except (KeyboardInterrupt, EOFError):
            send_message(sock, MSG_QUIT)
            running = False
            print(f"\n  {C.YELLOW}[Chat ended]{C.END}")
            break
    
    recv_thread.join(timeout=2)


def main():
    print_banner()
    
    # Load enrollment data
    helper_data, device_id, peer_pubkey = load_enrollment_data()
    
    # Determine server IP
    server_ip = SERVER_IP
    if len(sys.argv) > 1:
        server_ip = sys.argv[1]
    
    # Connect to Device A
    print(f"\n{C.YELLOW}[CLIENT] Connecting to Device A at {server_ip}:{SERVER_PORT}...{C.END}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    
    try:
        sock.connect((server_ip, SERVER_PORT))
    except (ConnectionRefusedError, socket.timeout) as e:
        print(f"{C.RED}ERROR: Could not connect to {server_ip}:{SERVER_PORT}")
        print(f"  {e}")
        print(f"  Make sure Device A is running first!{C.END}")
        sys.exit(1)
    
    sock.settimeout(None)
    print(f"  {C.GREEN}✓ Connected to Device A{C.END}")
    
    try:
        # ── Phase 1: PUF Authentication ──
        if not phase1_authenticate(sock, helper_data, device_id, peer_pubkey):
            print(f"{C.RED}ABORTING: Authentication failed{C.END}")
            sock.close()
            return
        
        # ── Phase 2: PLKG Key Agreement ──
        plkg_key = phase2_plkg(sock, server_ip)
        
        # ── Phase 3: PQC Key Exchange ──
        pqc_key = phase3_pqc(sock)
        
        # ── Phase 4: Key Derivation ──
        aes_key = phase4_derive_key(sock, plkg_key, pqc_key)
        
        # ── Phase 5: Encrypted Chat ──
        phase5_secure_chat(sock, aes_key)
        
    except Exception as e:
        print(f"\n{C.RED}ERROR: {e}{C.END}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        print(f"\n{C.YELLOW}[CLIENT] Disconnected{C.END}")


if __name__ == "__main__":
    main()
