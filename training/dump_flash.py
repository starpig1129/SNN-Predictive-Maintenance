"""
Read data collected to LittleFS flash back to a CSV file.

Usage
-----
  python dump_flash.py --port COM5 --output data/normal.csv

Steps
-----
  1.  Flash data-collection firmware (nodemcu-32s env).
  2.  Power ESP32 with motor OFF — green LED means ready and recording.
  3.  Start the motor; let it run as long as needed (USB can disconnect — that's fine).
  4.  Stop the motor and wait for it to spin down.
  5.  Run this script.  It sends 'd' over serial and saves the flash file.
"""

import argparse
import pathlib
import time

import serial


def main():
    parser = argparse.ArgumentParser(description="Dump LittleFS data from ESP32")
    parser.add_argument("--port",   required=True,            help="Serial port (e.g. COM5)")
    parser.add_argument("--output", default="data/dump.csv",  help="Output CSV path")
    parser.add_argument("--baud",   type=int, default=115200)
    args = parser.parse_args()

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {args.port} ...")
    with serial.Serial(args.port, args.baud, timeout=15) as ser:
        time.sleep(2)                    # let the serial driver settle
        ser.reset_input_buffer()

        print("Sending dump command ('d') ...")
        ser.write(b"d")

        # Wait for DATA_START marker
        deadline = time.time() + 15
        while time.time() < deadline:
            raw = ser.readline()
            line = raw.decode("utf-8", errors="ignore").strip()
            if line == "DATA_START":
                print("Receiving data ...")
                break
            if line:
                print(f"  esp32: {line}")
        else:
            print("ERROR: timed out waiting for DATA_START")
            return

        # Collect until DATA_END
        lines: list[str] = []
        while True:
            raw = ser.readline()
            line = raw.decode("utf-8", errors="ignore").strip()
            if line == "DATA_END":
                break
            if line:
                lines.append(line)

    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n_samples = max(0, len(lines) - 1)   # subtract header row
    print(f"Saved {n_samples} samples → {out}")


if __name__ == "__main__":
    main()
