# config.py — Shared configuration for both devices

# ============================================================
# NETWORK SETTINGS
# ============================================================
# Change SERVER_IP to Device A's IP address on your network
SERVER_IP = "ALICE_IP"      # Device A (Alice) IP — CHANGE THIS
SERVER_PORT = 5555               # TCP port for communication

# ============================================================
# SECURITY PARAMETERS
# ============================================================
PUF_RESPONSE_BITS = 2048         # PUF response size in bits
PUF_KEY_BYTES = 32               # Derived key size (256 bits)
PLKG_KEY_BYTES = 16              # PLKG key size (128 bits)
AES_KEY_BYTES = 32               # Final AES-256 key
NONCE_BYTES = 32                 # Challenge nonce size
AUTH_TIMEOUT_SECONDS = 30        # Max time for auth response

# KEM algorithm (Kyber)
KEM_ALGORITHM = "Kyber768"

USE_DAE = False

# ============================================================
# FILE PATHS
# ============================================================
ENROLLMENT_DIR = "enrollment_data"
HELPER_DATA_FILE = "enrollment_data/helper_data.json"
PEER_PUBKEY_FILE = "enrollment_data/peer_public_key.bin"
DEVICE_ID_FILE = "enrollment_data/device_id.txt"

# ============================================================
# PROTOCOL MESSAGE TYPES
# ============================================================
MSG_AUTH_CHALLENGE = b"AUTH_CHALLENGE"
MSG_AUTH_RESPONSE = b"AUTH_RESPONSE"
MSG_AUTH_OK = b"AUTH_OK"
MSG_AUTH_FAIL = b"AUTH_FAIL"
MSG_PLKG_START = b"PLKG_START"
MSG_PLKG_DATA = b"PLKG_DATA"
MSG_PLKG_DONE = b"PLKG_DONE"
MSG_PQC_PUBKEY = b"PQC_PUBKEY"
MSG_PQC_CIPHERTEXT = b"PQC_CT"
MSG_KEY_READY = b"KEY_READY"
MSG_CHAT = b"CHAT"
MSG_QUIT = b"QUIT"
