#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def extract_bytes(header_path: Path) -> bytes:
    data = bytearray()
    hex_re = re.compile(r"0x([0-9a-fA-F]{1,2})")
    dec_re = re.compile(r"\b([0-9]{1,3})\b")
    inside = False
    with header_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not inside:
                # Look for the opening brace of the initializer
                if '{' in line:
                    inside = True
                    # Trim everything before '{' for this line
                    line = line.split('{', 1)[1]
                else:
                    continue
            # Stop after closing brace
            if '}' in line:
                line = line.split('}', 1)[0]
                # We'll break after processing this part
                end_reached = True
            else:
                end_reached = False
            # Extract hex numbers first
            for hx in hex_re.findall(line):
                data.append(int(hx, 16))
            # Then decimals that are not part of hex notation
            # To avoid double counting, remove hex literals
            line_no_hex = re.sub(r"0x[0-9a-fA-F]{1,2}", " ", line)
            for dec in dec_re.findall(line_no_hex):
                val = int(dec, 10)
                if 0 <= val <= 255:
                    data.append(val)
            if end_reached:
                break
    return bytes(data)


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_model_from_header.py <input_header> <output_tflite>")
        sys.exit(1)
    header = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if not header.exists():
        print(f"Input header does not exist: {header}")
        sys.exit(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = extract_bytes(header)
    out.write_bytes(data)
    print(f"Wrote {len(data)} bytes to {out}")

if __name__ == "__main__":
    main()
