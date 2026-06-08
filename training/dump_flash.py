"""
Read data collected to LittleFS flash back to a CSV file.

Usage
-----
  # 步驟一：清除舊資料，準備新的錄製 session
  python dump_flash.py --port COM5 --clear

  # 步驟二：拔掉 USB，啟動馬達，讓它跑 5–10 分鐘，再停下來

  # 步驟三：重新插上 USB，下載資料
  python dump_flash.py --port COM5 --output data/normal.csv
"""

import argparse
import pathlib
import time

import serial


def _send_clear(ser: serial.Serial) -> bool:
    ser.write(b"c")
    deadline = time.time() + 5
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "OK:cleared":
            return True
        if line:
            print(f"  esp32: {line}")
    return False


def _send_dump(ser: serial.Serial) -> list:
    ser.write(b"d")
    deadline = time.time() + 15
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "DATA_START":
            print("Receiving data ...")
            break
        if line:
            print(f"  esp32: {line}")
    else:
        print("ERROR: timed out waiting for DATA_START")
        return []

    lines = []
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "DATA_END":
            break
        if line:
            lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Dump LittleFS data from ESP32")
    parser.add_argument("--port",   required=True,            help="Serial port (e.g. COM5)")
    parser.add_argument("--output", default="data/dump.csv",  help="Output CSV path")
    parser.add_argument("--baud",   type=int, default=115200)
    parser.add_argument("--clear",  action="store_true",
                        help="Clear existing flash data and start a new recording session")
    args = parser.parse_args()

    print(f"Connecting to {args.port} ...")
    with serial.Serial(args.port, args.baud, timeout=15) as ser:
        time.sleep(2)
        ser.reset_input_buffer()

        if args.clear:
            print("Clearing flash data ...")
            if _send_clear(ser):
                print("OK — flash cleared. Unplug USB, run the motor, then reconnect and run without --clear to download.")
            else:
                print("ERROR: no response to clear command")
            return

        lines = _send_dump(ser)

    if not lines:
        return

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n_samples = max(0, len(lines) - 1)
    print(f"Saved {n_samples} samples → {out}")


if __name__ == "__main__":
    main()
