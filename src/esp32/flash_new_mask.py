import subprocess, sys

def write_mask_to_firmware(mask_bits, src_path):
    with open(src_path) as f:
        content = f.read()
    # crude replace — better to use a template with a placeholder marker
    mask_str = ", ".join(str(b) for b in mask_bits)
    # locate and replace the bit_mask[] array block — recommend using a
    # marker like /*MASK_START*/ ... /*MASK_END*/ in your .c file for safety
    ...
    with open(src_path, "w") as f:
        f.write(content)

def build_and_flash(project_dir, port):
    subprocess.run(["idf.py", "-p", port, "build", "flash"], cwd=project_dir, check=True)
