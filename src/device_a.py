#!/usr/bin/env python3
# device_a.py — Device A (Alice / Server)
# ============================================================
# Run this on Raspberry Pi #1.
# This device acts as the SERVER — it listens for connections
# from Device B and manages the security handshake.
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

# ANSI colors for terminal output
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
    print(f"""{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════════╗
║          TRIPLE-LAYER SECURITY — DEVICE A (Alice)        ║
║              PUF + PLKG + PQC/AES-256-GCM                ║
╠══════════════════════════════════════════════════════════╣
║  Role:   SERVER (listening for Device B)                 ║
║  Layers: 1. PUF Authentication                           ║
║          2. PLKG Key Agreement                           ║
║          3. PQC (Kyber) Key Exchange                     ║
║          4. AES-256-GCM Encrypted Chat                   ║
╚══════════════════════════════════════════════════════════╝
{C.END}""")


def load_enrollment_data():
    """Load PUF helper data and peer's public key."""
    print(f"\n{C.YELLOW}[SETUP] Loading enrollment data...{C.END}")
    
    # Load helper data
    if not os.path.exists(HELPER_DATA_FILE):
        print(f"{C.RED}ERROR: No enrollment data found!")
        print(f"Run: python3 enrollment.py enroll pi-alice{C.END}")
        sys.exit(1)
    
    helper_data = load_helper_data(HELPER_DATA_FILE)
    print(f"  ✓ Helper data loaded")
    
    # Load device ID
    with open(DEVICE_ID_FILE, "r") as f:
        device_id = f.read().strip()
    print(f"  ✓ Device ID: {device_id}")
    
    # Load peer's public key
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
    
    Alice (server) flow:
    1. Receive challenge from Bob → sign with PUF key → send signature
    2. Send challenge to Bob → receive signature → verify
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 1: PUF AUTHENTICATION")
    print(f"{'='*60}{C.END}")
    
    # Re-derive PUF signing key
    print(f"  {C.CYAN}Extracting PUF response...{C.END}")
    # Read ESP32 hardware PUF root key from file
    try:
        with open('/home/pi_alice/alice_HS/debug_v2/alice_puf.txt', 'r') as f:
            puf_key_hex = f.read().strip()
        puf_key = bytes.fromhex(puf_key_hex)
        print(f"  [DEBUG] Loaded puf_key hex: {puf_key.hex()}")
        print(f"  [DEBUG] Loaded puf_key length: {len(puf_key)}")
        print(f"  {C.CYAN}Loading hardware PUF key from ESP32...{C.END}")
    except FileNotFoundError:
        print(f"  {C.RED}✗ ERROR: ESP32 root key not found!{C.END}")
        print(f"  {C.RED}  Run: python3 read_puf_key.py on this Pi first{C.END}")
        return False

    # Use fuzzy extractor to get stable key from noisy PUF response
    #fe = FuzzyExtractor()
    #puf_key_stable = fe.rep(puf_key, helper_data)
    sk, pk = derive_signing_keypair(puf_key)
    print(f"  {C.GREEN}✓ Hardware PUF key loaded, signing key derived{C.END}")
    # Export Alice's public key for Bob
    alice_pubkey_bytes = bytes(pk)
    with open('/home/pi_alice/alice_HS/debug_v2/alice_pubkey.txt', 'w') as f:
        f.write(alice_pubkey_bytes.hex())
    print(f"  [DEBUG] Exported Alice public key to /tmp/alice_pubkey.txt")
    print(f"  {C.GREEN}✓ PUF key recovered, signing key derived{C.END}")
    
    
    # Step 1: Receive Bob's challenge, sign it
    msg_type, challenge_from_bob = recv_message(sock)
    if msg_type != MSG_AUTH_CHALLENGE:
        raise ValueError(f"Protocol error: expected AUTH_CHALLENGE, got {msg_type}")
    signature = sign_message(sk, challenge_from_bob)
    send_message(sock, MSG_AUTH_RESPONSE, 
                 device_id.encode() + b"|" + signature)
    print(f"  {C.GREEN}✓ Signed Bob's challenge → sent signature{C.END}")
    
    # Step 2: Send our challenge to Bob
    our_challenge = os.urandom(NONCE_BYTES)
    send_message(sock, MSG_AUTH_CHALLENGE, our_challenge)
    
    # Receive Bob's signature
    msg_type, auth_data = recv_message(sock)
    if msg_type != MSG_AUTH_RESPONSE:
        raise ValueError(f"Protocol error: expected AUTH_RESPONSE, got {msg_type}")
    peer_id, peer_sig = auth_data.split(b"|", 1)
    
    # Verify Bob's signature
    valid = verify_signature(peer_pubkey, our_challenge, peer_sig)
    
    if valid:
        send_message(sock, MSG_AUTH_OK)
        print(f"  {C.GREEN}{C.BOLD}✓ MUTUAL AUTHENTICATION SUCCESSFUL{C.END}")
        print(f"  {C.GREEN}  Peer device: {peer_id.decode()}{C.END}")
        return True
    else:
        send_message(sock, MSG_AUTH_FAIL)
        print(f"  {C.RED}✗ AUTHENTICATION FAILED — peer signature invalid!{C.END}")
        return False


def phase2_plkg(sock, peer_ip):
    """
    Phase 2: PLKG Key Agreement
    
    Both devices measure the channel simultaneously and derive
    a shared 128-bit key from the wireless channel.
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 2: PLKG KEY AGREEMENT")
    print(f"{'='*60}{C.END}")
    
    plkg = PLKGKeyGenerator("alice", peer_ip)
    
    # Signal Bob to start measuring
    send_message(sock, MSG_PLKG_START)
    
    # Measure channel
    plkg.measure_channel(num_samples=50)
    plkg.quantize()
    
    # Wait for Bob's ready signal
    msg_type, _ = recv_message(sock)
    if msg_type != MSG_PLKG_DONE:
        raise ValueError(f"Protocol error: expected PLKG_DONE, got {msg_type}")    
    # Exchange syndromes for reconciliation
    our_syndrome = plkg.get_syndrome()
    send_message(sock, MSG_PLKG_DATA, our_syndrome)
    
    msg_type, peer_syndrome = recv_message(sock)
    if msg_type != MSG_PLKG_DATA:
        raise ValueError(f"Protocol error: expected PLKG_DATA, got {msg_type}")
    
    # Reconcile and amplify
    plkg.reconcile(peer_syndrome)
    session_ctx = os.urandom(16)
    send_message(sock, MSG_PLKG_DATA, session_ctx)
    plkg_key = plkg.amplify(session_ctx)
    
    print(f"  {C.GREEN}{C.BOLD}✓ PLKG key generated: {plkg_key.hex()}{C.END}")
    return plkg_key


def phase3_pqc(sock):
    """
    Phase 3: PQC Key Exchange (Kyber)
    
    Alice generates Kyber key pair, sends public key to Bob.
    Bob encapsulates and sends back ciphertext.
    Alice decapsulates to get shared secret.
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 3: PQC KEY EXCHANGE (KYBER)")
    print(f"{'='*60}{C.END}")
    
    pqc = PQCKeyExchange()
    
    # Generate key pair
    print(f"  {C.CYAN}Generating Kyber-768 key pair...{C.END}")
    public_key = pqc.generate_keypair()
    
    # Send public key to Bob
    send_message(sock, MSG_PQC_PUBKEY, public_key)
    print(f"  {C.GREEN}✓ Public key sent to Bob ({len(public_key)} bytes){C.END}")
    
    # Receive ciphertext from Bob
    msg_type, ciphertext = recv_message(sock)
    if msg_type != MSG_PQC_CIPHERTEXT:
        raise ValueError(f"Protocol error: expected PQC_CIPHERTEXT, got {msg_type}")
    print(f"  {C.CYAN}Received ciphertext from Bob ({len(ciphertext)} bytes){C.END}")
    
    # Decapsulate to get shared secret
    print(f"  {C.CYAN}Decapsulating...{C.END}")
    pqc_key = pqc.decapsulate(ciphertext)
    
    print(f"  {C.GREEN}{C.BOLD}✓ PQC shared secret: {pqc_key.hex()[:32]}...{C.END}")
    return pqc_key


def phase4_derive_key(sock, plkg_key, pqc_key):
    """
    Phase 4: Derive final AES-256 key from PLKG + PQC via HKDF.
    Alice generates session_id and sends it to Bob so both use the same value.
    """
    print(f"\n{C.BOLD}{'='*60}")
    print(f"  PHASE 4: KEY DERIVATION (HKDF)")
    print(f"{'='*60}{C.END}")
    
    # Session ID derived from shared values + random salt
    # Alice generates and sends to Bob to ensure both match
    salt = os.urandom(16)
    session_id = hashlib.sha256(
        plkg_key + pqc_key + salt
    ).digest()
    send_message(sock, MSG_KEY_READY, salt)
    
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
    # Add sequence number tracking
    alice_last_seq = 0  # Track last received sequence
    alice_send_seq = 0  # Track outgoing sequence
    alice_replay_counter=0
    REPLAY_THRESHOLD=3
    cipher = AESCipher(aes_key)
    running = True
    
    def receive_messages():
        """Background thread to receive and decrypt messages."""
        nonlocal running, alice_last_seq, alice_replay_counter
        while running:
            try:
                msg_type, encrypted_payload = recv_message(sock)
                
                if msg_type == MSG_QUIT:
                    print(f"\n  {C.YELLOW}[Bob has left the chat]{C.END}")
                    running = False
                    break
                
                if msg_type == MSG_CHAT:
                    try:
                        plaintext = cipher.decrypt_text(encrypted_payload)
                        # Parse message with sequence number
                        msg_obj = json.loads(plaintext)
                        seq = msg_obj.get("seq", 0)
                        text = msg_obj.get("text", "")
                        # Check sequence number
                        if seq <= alice_last_seq:
                            alice_replay_counter += 1
                            print(f"\r  {C.RED}✗ REPLAY DETECTED (seq {seq} ≤ {alice_last_seq}){C.END}")

                            if alice_replay_counter >= REPLAY_THRESHOLD:
                                print(f"  {C.RED}{C.BOLD}✗ REPLAY THRESHOLD EXCEEDED — TERMINATING SESSION{C.END}")
                                running = False
                                try:
                                    send_message(sock, MSG_QUIT)
                                except:
                                    pass
                                break

                            print(f"  {C.GREEN}[You]{C.END} ", end="", flush=True)
                            continue

                        alice_last_seq = seq
                        print(f"\r  {C.BLUE}{C.BOLD}[Bob]{C.END} {text}")
                        print(f"  {C.GREEN}[You]{C.END} ", end="", flush=True)
                    except ValueError as e:
                        print(f"\n  {C.RED}{C.BOLD}⚠ TAMPERED MESSAGE DETECTED: {e}{C.END}")
                        print(f"  {C.RED}{C.BOLD}✗ INTEGRITY VIOLATION — TERMINATING SESSION IMMEDIATELY{C.END}")
                        running = False
                        try:
                            send_message(sock, MSG_QUIT)
                        except:
                            pass
                        break
            except ConnectionError:
                print(f"\n  {C.RED}[Connection lost]{C.END}")
                running = False
                break
            except Exception as e:
                if running:
                    print(f"\n  {C.RED}[Error: {e}]{C.END}")
                break
    
    # Synchronize ready state BEFORE starting receiver thread
    msg_type, _ = recv_message(sock)  # Wait for Bob's ready
    send_message(sock, MSG_KEY_READY)  # Confirm to Bob
    
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
            # Increment sequence number
            alice_send_seq += 1
            # Create message with sequence
            msg_obj = {
                "seq": alice_send_seq,
                "text": user_input
            }
            
            # Encrypt and send
            encrypted = cipher.encrypt_text(json.dumps(msg_obj))
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
    
    # Start server
    print(f"\n{C.YELLOW}[SERVER] Starting on {SERVER_IP}:{SERVER_PORT}...{C.END}")
    
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_sock.bind(("0.0.0.0", SERVER_PORT))
    except OSError as e:
        print(f"{C.RED}ERROR: Could not bind to port {SERVER_PORT}: {e}{C.END}")
        print(f"Try: sudo lsof -i :{SERVER_PORT}  (to see what's using it)")
        sys.exit(1)
    
    server_sock.listen(1)
    print(f"  {C.GREEN}✓ Listening for Device B connections...{C.END}")
    print(f"  {C.CYAN}(Waiting for Bob to connect){C.END}\n")
    
    conn, addr = server_sock.accept()
    peer_ip = addr[0]
    print(f"  {C.GREEN}✓ Device B connected from {peer_ip}:{addr[1]}{C.END}")
    
    try:
        # ── Phase 1: PUF Authentication ──
        if not phase1_authenticate(conn, helper_data, device_id, peer_pubkey):
            print(f"{C.RED}ABORTING: Authentication failed{C.END}")
            conn.close()
            return
        
        # ── Phase 2: PLKG Key Agreement ──
        plkg_key = phase2_plkg(conn, peer_ip)
        
        # ── Phase 3: PQC Key Exchange ──
        pqc_key = phase3_pqc(conn)
        
        # ── Phase 4: Key Derivation ──
        aes_key = phase4_derive_key(conn, plkg_key, pqc_key)
        
        # ── Phase 5: Encrypted Chat ──
        phase5_secure_chat(conn, aes_key)
        
    except Exception as e:
        print(f"\n{C.RED}ERROR: {e}{C.END}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        server_sock.close()
        print(f"\n{C.YELLOW}[SERVER] Shutdown complete{C.END}")


if __name__ == "__main__":
    main()
