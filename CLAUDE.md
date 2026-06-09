# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predictive maintenance system using Spiking Neural Networks (SNNs). Three-tier architecture:
- **Firmware** (ESP32/NodeMCU-32S): Reads MPU6050 X-axis acceleration at 200 Hz over I2C, writes to LittleFS flash or streams CSV over serial. Two compile-time modes: data collection (default) and on-device SNN inference (`-DINFERENCE_MODE`).
- **Training** (Python/PyTorch + snnTorch): Collects sensor data → delta modulation → trains streaming SNN → exports weights to C header.
- **Dashboard** (PyQt5 + pyqtgraph): Real-time waveform + fault logit visualisation, connects to ESP32 in inference mode.

## Commands

All firmware commands must be run from inside the `firmware/` directory. On Windows, `pio` is not on PATH; use the full path `C:\Users\<user>\.platformio\penv\Scripts\platformio.exe`.

### Firmware
```bash
cd firmware
C:\Users\james\.platformio\penv\Scripts\platformio.exe run -e nodemcu-32s                 # Build only
C:\Users\james\.platformio\penv\Scripts\platformio.exe run -t upload -e nodemcu-32s       # Flash data collection
C:\Users\james\.platformio\penv\Scripts\platformio.exe run -t upload -e nodemcu-32s-infer # Flash inference
C:\Users\james\.platformio\penv\Scripts\platformio.exe device monitor -e nodemcu-32s-infer --baud 115200
```

### Data Collection (use dump_flash.py — NOT logger.py for motor-on data)
```bash
cd training

# Step 1 — clear flash and start a new recording session
python dump_flash.py --port COM5 --clear

# Step 2 — unplug USB, power ESP32 from power bank, run motor for 5–10 min, then stop motor

# Step 3 — reconnect USB, download data
python dump_flash.py --port COM5 --output data/normal.csv   # motor running normally
python dump_flash.py --port COM5 --output data/idle.csv     # motor stopped (no EMI, can use logger.py)
```

**Why not logger.py for motor-on data**: Motor EMI causes ~87% I2C failure rate, causing `mpu.getEvent()` to block repeatedly. `logger.py` (which requires real-time serial) freezes. The flash-based approach bypasses this entirely.

### Training
```bash
cd training
# Motor ON/OFF detection (current working approach)
python train_snn.py --mode streaming --fault-file idle.csv --normal-file normal.csv --chunk-size 50 --decision-interval 50
python export_weights.py   # writes firmware/include/snn_weights.h

# With CWRU bearing dataset (NOT recommended — see Known Issues)
python train_snn.py --mode streaming --dataset cwru --cwru-normal data/cwru/97.mat --cwru-fault data/cwru/105.mat --delta-thresh 0.05
```

### Dashboard
```bash
cd dashboard
python main.py    # Requires ESP32 running nodemcu-32s-infer firmware
                  # Select COM port in dropdown, click Connect
```

## Architecture

### End-to-End Data Flow
```
ESP32 (MPU6050, 200 Hz, I2C SDA=21/SCL=22, clock=100 kHz)
  → LittleFS flash (data collection mode, no USB needed)
    → dump_flash.py ('d' command dumps CSV over serial)
      → data/idle.csv  +  data/normal.csv
        → train_snn.py  (delta modulation → StreamingVibrationSNN → BPTT)
          → model.pth
            → export_weights.py
              → firmware/include/snn_weights.h
                → nodemcu-32s-infer firmware (streaming forward pass on ESP32)
                  → Serial  →  dashboard/main.py
```

### SNN Model — Streaming Mode (`training/train_snn.py`)
- **Input encoding**: delta modulation — spike=1 when `|Δaccel_z| > DELTA_THRESH` (default 0.10 m/s²). One spike per sample (not a fixed window).
- **Network**: `Linear(1→32) + Leaky` → `Linear(32→2)`. No bias terms. Beta is learned.
- **Training**: Streaming BPTT on chunks of `CHUNK_SIZE=50` samples. `mem1` initialised to zero at the start of each chunk.
- **Key hyperparameters**: `HIDDEN=32`, `CHUNK_SIZE=50`, `DECISION_INTERVAL=50`, `BETA≈0.83` (learned), `THRESHOLD=0.1`, `DELTA_THRESH=0.10`.
- **Labels**: fault/idle = 0, normal running = 1.

### Firmware Inference Engine (`firmware/src/main.cpp`)
- Streaming: one delta-encoded spike per sample, processed immediately.
- `mem1[32]` is reset to zero after each decision window (`memset` — matches training chunk initialisation).
- `out_acc[2]` accumulates FC2 logits over `SNN_DECISION_INTERVAL` samples, then resets.
- **Decision logic** (two-stage):
  1. If `hidden_rate < SNN_IDLE_RATE_THRESH (0.15)` → anomaly (motor stopped).
  2. Else: anomaly if `out_acc[0] >= out_acc[1]` (fault logit ≥ normal logit).
- `hidden_rate = total hidden neuron firings / (DECISION_INTERVAL × HIDDEN)` — reflects vibration intensity.
- I2C failures are handled: skip spike computation, do not update `prev_az`, call `i2c_recover()` after 10 consecutive failures.
- **LED states**: green + blue blinking = NORMAL; red rapid flash = ANOMALY.

### hidden_rate Explained
`hidden_rate` measures the average firing rate of the 32 hidden LIF neurons per sample within the current decision window. It reflects vibration intensity after SNN processing:

| State | hidden_rate | Reason |
|-------|-------------|--------|
| Motor stopped | ~0.07–0.14 | Almost no vibration → few input spikes → few neuron firings |
| Motor running | ~0.30–0.40 | Continuous vibration → many input spikes → many neuron firings |

The 0.15 threshold cleanly separates these two states and is more robust than relying on the FC2 logits alone.

### Serial Protocol (Inference Mode)
Two interleaved line types:
```
<timestamp_ms>,<accel_z>                            ← raw sample, every 5 ms (skipped on I2C failure)
INF,<hidden_rate>,<fault_acc>,<status>,<fault_acc>  ← decision result, every SNN_DECISION_INTERVAL samples
```
`status` is `0` (normal) or `1` (anomaly). Dashboard parses both prefixes.

### Hardware
- I2C: SDA=21, SCL=22, clock forced to 100 kHz (motor EMI tolerance), `Wire.setTimeOut(15)` (prevents >15 ms hangs)
- Status LEDs: R=GPIO12, G=GPIO13, B=GPIO14 (common-cathode RGB, active HIGH)
- Sensor: ±4G range, hardware 94 Hz LPF (`MPU6050_BAND_94_HZ`), reads `acceleration.x`
- Motor power is **fully isolated** from ESP32 — never share ground or supply
- Data collection firmware preserves flash data across resets. Send `'c'` over serial (via `dump_flash.py --clear`) to start a new session; send `'d'` to dump.

## Key Files

| File | Purpose |
|------|---------|
| `firmware/src/main.cpp` | Sensor sampling, I2C recovery, streaming SNN forward pass, LED logic |
| `firmware/include/snn_weights.h` | Auto-generated — do NOT edit; produced by `export_weights.py`. Contains `SNN_IDLE_RATE_THRESH`. |
| `firmware/platformio.ini` | Two envs: `nodemcu-32s` (collection) and `nodemcu-32s-infer` |
| `training/dump_flash.py` | Sends `'c'` (clear) or `'d'` (dump) to ESP32 flash; use instead of logger.py for motor-on data |
| `training/logger.py` | Serial → CSV (real-time). Only reliable when motor is OFF (no EMI). |
| `training/train_snn.py` | Delta modulation, `StreamingVibrationSNN` definition, training loop. Supports `--fault-file` / `--normal-file` to select training CSVs. |
| `training/export_weights.py` | Loads `model.pth`, writes C header with float arrays and config macros |
| `dashboard/main.py` | PyQt5 + pyqtgraph; uses `queue.Queue` for raw samples (avoids 200 Hz Qt signal overhead); `SerialReader` QThread |

## Known Issues & Learnings

### Motor EMI (Critical)
Motor operation causes ~87% I2C failure rate, reducing effective sample rate from 200 Hz to ~25 Hz. Consequences:
- `logger.py` freezes — use `dump_flash.py` instead
- Delta values in training data are computed across ~40 ms gaps (not 5 ms)
- The same sparse pattern appears at inference, so training/inference distributions match
- **Hardware fix**: 100 nF ceramic capacitor across motor terminals eliminates EMI at source

### CWRU Dataset Does Not Work at 200 Hz
CWRU bearing fault data is sampled at 12 kHz. After 60× downsampling to 200 Hz, fault-characteristic frequencies (ball-pass frequency 105–162 Hz) are completely lost. The model learns nothing useful and defaults to predicting one class. Do not use CWRU for this hardware setup.

### Training/Inference State Mismatch (Fixed)
The streaming SNN training initialises `mem1=0` for each chunk, but early inference firmware never reset `mem1`. After the motor stopped, residual membrane charge caused continued hidden neuron firing, making the system report NORMAL when the motor was stopped. **Fixed** by `memset(mem1, 0, sizeof(mem1))` after each decision window in the inference loop.

### I2C Failure in Inference Loop (Fixed)
Early inference firmware called `mpu.getEvent()` without checking the return value. On I2C failure, `az=0`, which created a large spurious delta spike (e.g., `|0 – 9.8| >> threshold`), inflating the fault logit. **Fixed** by checking the return value and skipping spike computation on failure.

### idle.csv Distribution vs. Deployment Environment
`idle.csv` collected in a quiet environment shows near-zero deltas. At deployment, the motor structure transmits ambient vibrations even when stopped, producing hidden_rate ~0.07–0.14 instead of ~0. The `SNN_IDLE_RATE_THRESH=0.15` rule handles this gap, but if the environment changes significantly, the threshold may need tuning.

### Anomaly Detection (Touching Fan) Did Not Work
Attempted to train normal vs. hand-touching-fan anomaly detection. Failed because:
1. Original `anomaly.csv` was collected without the motor running (logger.py froze with motor on), so the anomaly pattern (low vibration + touch) did not match inference conditions (high vibration + touch).
2. Only ~29 training chunks — insufficient for learning subtle vibration differences.
3. After resampling CWRU data fails, this remains an unsolved problem for this hardware setup.

## Notes

- `.gitignore` excludes `data/*.csv`, `*.pth`, `.pio/` — never commit raw sensor data or trained weights.
- `snn_weights.h` does not exist until `export_weights.py` is run; the `nodemcu-32s-infer` build will fail without it.
- The serial port is hardcoded as `COM5` in `platformio.ini` — update both `upload_port` and `monitor_port` if the device enumerates differently. Dashboard auto-discovers available ports via dropdown.
- When changing `--chunk-size` or `--decision-interval` during training, always re-run `export_weights.py` and reflash the inference firmware — these values must match.
- `training/notebooks/` is reserved for exploratory analysis; keep production logic in `.py` scripts.
