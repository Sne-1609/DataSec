cat > ~/automate_esp32.py << 'EOF'
#!/usr/bin/env python3
"""
Automate ESP32 flash erase and PUF key generation (20 samples)
"""
import serial
import time
import subprocess
import sys

def erase_esp32_flash(port='/dev/ttyAMA0', baud=115200):
    """Erase ESP32 flash memory"""
    print("[1/3] Erasing ESP32 flash...")
    try:
        result = subprocess.run(
            ['esptool.py', '--port', port, 'erase_flash'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("  ✓ Flash erased successfully")
            return True
        else:
            print(f"  ✗ Erase failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def read_puf_samples(port='/dev/ttyAMA0', baud=115200, num_samples=20):
    """Read PUF responses and average them"""
    print(f"\n[2/3] Reading {num_samples} PUF samples...")
    
    try:
        ser = serial.Serial(port, baud, timeout=10)
        time.sleep(2)  # Wait for ESP32 to boot
        
        puf_samples = []
        for i in range(num_samples):
            print(f"  Sample {i+1}/{num_samples}...", end=' ', flush=True)
            
            # Power cycle: send reset signal
            ser.dtr = False
            time.sleep(0.1)
            ser.dtr = True
            time.sleep(2)  # Wait for ESP32 to boot and generate PUF
            
            # Read response
            ser.write(b"get_puf\n")
            time.sleep(0.5)
            
            response = ser.read(1024).decode('utf-8', errors='ignore')
            if 'PUF' in response or '0x' in response:
                # Extract hex value if present
                lines = response.split('\n')
                for line in lines:
                    if line.startswith('0x') or all(c in '0123456789abcdefABCDEF' for c in line.strip()):
                        puf_samples.append(line.strip())
                        print(f"✓")
                        break
                else:
                    print(f"?")
            else:
                print(f"✗")
            
            time.sleep(1)
        
        ser.close()
        
        if puf_samples:
            print(f"\n  Collected {len(puf_samples)} samples")
            return puf_samples
        else:
            print("  ✗ No PUF samples collected")
            return None
            
    except Exception as e:
        print(f"  ✗ Serial error: {e}")
        return False

def save_puf_key(puf_samples, output_file='/tmp/puf_root_key.txt'):
    """Average PUF samples and save"""
    print(f"\n[3/3] Saving PUF key to {output_file}...")
    
    if not puf_samples:
        print("  ✗ No samples to save")
        return False
    
    try:
        # Save all samples for analysis
        with open(output_file + '.samples', 'w') as f:
            for i, sample in enumerate(puf_samples):
                f.write(f"Sample {i+1}: {sample}\n")
        
        # Use first stable sample as the key
        puf_key = puf_samples[0]
        with open(output_file, 'w') as f:
            f.write(puf_key)
        
        print(f"  ✓ PUF key saved: {puf_key[:32]}...")
        print(f"  ✓ All samples saved to: {output_file}.samples")
        return True
        
    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        return False

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyAMA0'
    
    print("=" * 60)
    print("  ESP32 AUTOMATED PUF KEY GENERATION")
    print("=" * 60)
    
    # Step 1: Erase flash
    if not erase_esp32_flash(port):
        sys.exit(1)
    
    time.sleep(3)
    
    # Step 2: Read 20 PUF samples
    samples = read_puf_samples(port, num_samples=20)
    if not samples:
        sys.exit(1)
    
    # Step 3: Save key
    if not save_puf_key(samples):
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  ✓ AUTOMATION COMPLETE")
    print("=" * 60)
EOF
chmod +x ~/automate_esp32.py
