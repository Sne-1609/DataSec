# DataSec

## Hardware-Assisted Secure Communication Framework

DataSec is a hardware-assisted security framework that combines device-specific
PUF entropy, physical-layer key generation, post-quantum cryptography,
cryptographic key derivation, authenticated encryption, and side-channel
monitoring.

The system is designed for secure communication between two embedded devices,
referred to as **Alice** and **Bob**.

The current implementation integrates:

- SRAM-based Physical Unclonable Function (PUF)
- Hamming (7,4) Error Correcting Code
- SHA-256 hardware-root-key derivation
- Physical Layer Key Generation (PLKG)
- ML-KEM-768 / Kyber-based post-quantum key exchange
- HKDF-based session-key derivation
- AES-256-GCM authenticated encryption
- INA219-based power monitoring
- Machine-learning-based Side-Channel Analysis (SCA)
- Denoising Autoencoder (DAE) experimentation
- ESP32-based PUF acquisition and automation
- Security and attack-testing modules

---

## 1. System Overview

The framework combines multiple independent sources of security rather than
depending on a single cryptographic mechanism.

At a high level:

```text
              ┌──────────────────────┐
              │      ESP32 PUF       │
              │  SRAM Startup Data   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Stability Analysis   │
              │ + Stable-Bit Mask    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Hamming (7,4) ECC  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │      SHA-256         │
              │  PUF Root Key        │
              └──────────┬───────────┘
                         │
                         ├───────────────────┐
                         │                   │
                         ▼                   ▼
                ┌────────────────┐   ┌─────────────────┐
                │      PLKG      │   │    ML-KEM-768   │
                │ Channel Entropy│   │   PQC Exchange  │
                └───────┬────────┘   └────────┬────────┘
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                         ┌──────────────────┐
                         │      HKDF        │
                         │ Session Key      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   AES-256-GCM    │
                         │ Authenticated    │
                         │ Encryption       │
                         └──────────────────┘

              ┌─────────────────────────────┐
              │ INA219 Power Monitoring     │
              │          ↓                  │
              │ SCA Feature Extraction      │
              │          ↓                  │
              │ ML Classifier               │
              │          ↓                  │
              │ Leakage Detection / Abort   │
              └─────────────────────────────┘
