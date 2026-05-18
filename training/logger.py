import serial
import csv
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="ESP32 Serial Data Logger")
    parser.add_argument("--port", required=True, help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--output", default="data/data.csv", help="Output CSV filename (default: data/data.csv)")
    args = parser.parse_args()

    count = 0

    with serial.Serial(args.port, baudrate=115200, timeout=1) as ser, \
         open(args.output, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_ms", "accel_z"])

        for _ in range(5):
            ser.readline()

        print(f"開始記錄，輸出至 {args.output}，按 Ctrl+C 停止...")

        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    writer.writerow(parts)
                    csvfile.flush()
                    count += 1
        except KeyboardInterrupt:
            pass

    print(f"紀錄結束，共收集了 {count} 筆資料")

if __name__ == "__main__":
    main()
