# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predictive maintenance system using Spiking Neural Networks (SNNs). Three-tier architecture:
- **Firmware** (ESP32/NodeMCU-32S): Reads MPU6050 Z-axis acceleration at 200 Hz over I2C, streams CSV over serial. Two compile-time modes: data collection (default) and on-device SNN inference (`-DINFERENCE_MODE`).
- **Training** (Python/PyTorch + snnTorch): Collects sensor data → delta modulation → trains SNN → exports weights to C header.
- **Dashboard** (PyQt5 + pyqtgraph): Real-time waveform + membrane potential visualisation, connects to ESP32 in inference mode.

## Commands

All firmware commands must be run from inside the `firmware/` directory. On Windows, `pio` may not be on PATH; use the full path `C:\Users\<user>\.platformio\penv\Scripts\platformio.exe`.

### Firmware
```bash
cd firmware
pio run -e nodemcu-32s                    # Build (data collection mode)
pio run -t upload -e nodemcu-32s          # Flash data collection firmware
pio run -t upload -e nodemcu-32s-infer    # Flash inference firmware (needs snn_weights.h first)
pio device monitor -e nodemcu-32s --baud 115200
```

### Training
```bash
cd training
pip install -r requirements.txt
python logger.py --port COM5 --output data/normal.csv   # Collect normal data
python logger.py --port COM5 --output data/anomaly.csv  # Collect anomaly data
python train_snn.py                                      # Train; saves model.pth
python export_weights.py                                 # Writes firmware/include/snn_weights.h
```

### Dashboard
```bash
cd dashboard
pip install -r requirements.txt
python main.py    # Requires ESP32 running nodemcu-32s-infer firmware
```

## Architecture

### End-to-End Data Flow
```
ESP32 (MPU6050, 200 Hz, I2C SDA=21/SCL=22, clock=100 kHz)
  → Serial 115200 baud
    → training/logger.py  (IIR LPF α≈0.747, jerk filter in firmware)
      → data/normal.csv  +  data/anomaly.csv
        → train_snn.py  (delta modulation → VibrationSNN → BPTT)
          → model.pth
            → export_weights.py
              → firmware/include/snn_weights.h
                → nodemcu-32s-infer firmware (forward pass on ESP32)
                  → Serial  →  dashboard/main.py
```

### SNN Model (`training/train_snn.py`)
- **Input encoding**: delta modulation — spike=1 when `|Δaccel_z| > DELTA_THRESH` (default 0.10 m/s²). Produces a binary vector of length `WINDOW=128`.
- **Network**: `Linear(128→32) + Leaky` → `Linear(32→2) + Leaky`. No bias terms (simplifies C export). Betas are learned.
- **Training**: rate-coded BPTT — same binary window presented `T_STEPS=16` times; output spike counts classify normal(0) / anomaly(1).
- **Key hyperparameters**: `WINDOW=128`, `HIDDEN=32`, `T_STEPS=16`, `BETA=0.9`, `THRESHOLD=1.0`, `DELTA_THRESH=0.10`.

### Firmware Inference Engine (`firmware/src/main.cpp`)
- Ring buffer accumulates 128 samples; full buffer triggers delta encode → forward pass.
- FC1 uses binary inputs: only adds weight columns where spike=1 (no multiplies).
- LIF soft-reset: `mem -= threshold` on spike (not zeroed).
- Anomaly declared when anomaly output neuron fires `> T_STEPS × 0.5` times per window.
- **LED states**: green steady = normal idle; green + blue blinking (rate ∝ hidden spike rate) = normal active; red rapid flash = anomaly.

### Serial Protocol (Inference Mode)
Two interleaved line types:
```
<timestamp_ms>,<accel_z>                        ← raw sample, every 5 ms
INF,<hidden_rate>,<anomaly_spk>,<status>,<mem2> ← inference result, every 128 samples
```
`status` is `0` (normal) or `1` (anomaly). Dashboard parses both prefixes.

### Hardware
- I2C: SDA=21, SCL=22, clock forced to 100 kHz (motor EMI tolerance)
- Firmware jerk filter: discards readings where `|Δaccel_z| > 15 m/s²` between consecutive 5 ms samples
- Status LEDs: R=GPIO12, G=GPIO13, B=GPIO14 (common-cathode RGB, active HIGH)
- Sensor: ±4G range, hardware 94 Hz LPF (`MPU6050_BAND_94_HZ`)
- Motor power is **fully isolated** from ESP32 — never share ground or supply

### Known Hardware Issue
Motor EMI causes periodic I2C Error 263 (timeout) from the Wire library. The jerk filter in firmware suppresses the resulting outlier readings. Adding a 100 nF ceramic capacitor across the motor terminals eliminates the EMI at the source.

## Key Files

| File | Purpose |
|------|---------|
| `firmware/src/main.cpp` | Sensor sampling, jerk filter, SNN forward pass, LED logic |
| `firmware/include/snn_weights.h` | Auto-generated — do not edit; produced by `export_weights.py` |
| `firmware/platformio.ini` | Two envs: `nodemcu-32s` (collection) and `nodemcu-32s-infer` |
| `training/logger.py` | Serial → CSV with IIR LPF; use `--no-lpf` to disable |
| `training/train_snn.py` | Delta modulation, `VibrationSNN` definition, training loop |
| `training/export_weights.py` | Loads `model.pth`, writes C header with float arrays |
| `dashboard/main.py` | PyQt5 + pyqtgraph; `SerialReader` QThread parses firmware output |

## Notes

- `.gitignore` excludes `data/*.csv`, `*.pth`, `.pio/` — never commit raw sensor data or trained weights.
- `snn_weights.h` does not exist until `export_weights.py` is run; the `nodemcu-32s-infer` build will fail without it.
- `training/notebooks/` is reserved for exploratory analysis; keep production logic in `.py` scripts.
- The serial port is hardcoded as `COM5` in `platformio.ini` — update both `upload_port` and `monitor_port` if the device enumerates differently.
