import serial
import csv
import argparse
import math

# First-order IIR low-pass filter — forward-Euler discretisation of RC circuit.
# At fs=200 Hz, fc=94 Hz: α ≈ 0.747  (close to pass-through; hardware MPU6050
# DLPF already does the heavy lifting; this rounds off any ADC quantisation spikes.)
_FS = 200.0
_FC = 94.0
_LPF_ALPHA = 1.0 / (1.0 + _FS / (2.0 * math.pi * _FC))  # ≈ 0.747


class LowPassFilter:
    def __init__(self, alpha: float = _LPF_ALPHA):
        self._alpha = alpha
        self._y = None

    def __call__(self, x: float) -> float:
        if self._y is None:
            self._y = x
        self._y = self._alpha * x + (1.0 - self._alpha) * self._y
        return self._y


def main():
    parser = argparse.ArgumentParser(description="ESP32 Serial Data Logger")
    parser.add_argument("--port",   required=True,          help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--output", default="data/data.csv", help="Output CSV filename")
    parser.add_argument("--no-lpf", action="store_true",    help="Disable software low-pass filter")
    args = parser.parse_args()

    lpf = LowPassFilter() if not args.no_lpf else None
    count = 0

    with serial.Serial(args.port, baudrate=115200, timeout=1) as ser, \
         open(args.output, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_ms", "accel_z"])

        for _ in range(5):          # flush startup noise
            ser.readline()

        print(f"開始記錄，輸出至 {args.output}，按 Ctrl+C 停止...")

        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("INF"):
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        ts  = parts[0]
                        val = float(parts[1])
                        if lpf is not None:
                            val = lpf(val)
                        writer.writerow([ts, f"{val:.4f}"])
                        csvfile.flush()
                        count += 1
                    except ValueError:
                        pass
        except KeyboardInterrupt:
            pass

    print(f"紀錄結束，共收集了 {count} 筆資料")


if __name__ == "__main__":
    main()
