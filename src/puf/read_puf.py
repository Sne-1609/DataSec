#!/usr/bin/env python3
import serial
import time
import re

# Open the UART connection
ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
time.sleep(0.5)

print("Waiting for ESP32 root key...")

root_key = None
while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"[ESP32] {line}")
            
            # Look for the root key line
            if "KEY:" in line:
                match = re.search(r'KEY:([0-9A-Fa-f]{64})', line)
                if match:
                    root_key = match.group(1)
                    print(f"\n✓ Captured root key: {root_key}\n")
                    
                    # Save to file
                    with open('/home/pi_alice/alice_HS/debug_v2/alice_puf.txt', 'w') as f:
                        f.write(root_key)
                    print("Root key saved to /home/alice_HS/debug_v2/alice_puf.txt")
                    
                    # Exit (or keep listening for next boot)
                    break
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

ser.close()
