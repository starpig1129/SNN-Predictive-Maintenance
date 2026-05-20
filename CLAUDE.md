# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predictive maintenance system using Spiking Neural Networks (SNNs). Three-tier architecture:
- **Firmware** (ESP32/NodeMCU-32S): Reads MPU6050 Z-axis acceleration at 200 Hz, streams CSV over serial
- **Training** (Python/PyTorch + snnTorch): Collects sensor data, trains SNN, exports weights to C header
- **Dashboard** (PyQt5 + pyqtgraph): Real-time visualization and SNN inference monitoring

Most training and dashboard code is still scaffolded (placeholder stubs). The implemented pieces are `firmware/src/main.cpp` and `training/logger.py`.

## Commands

### Firmware (PlatformIO required)
```bash
pio run -e nodemcu-32s          # Build
pio run -t upload -e nodemcu-32s # Flash to device
pio device monitor -b 115200    # Serial monitor
pio test -e nodemcu-32s         # Run firmware tests
```

### Training
```bash
cd training
pip install -r requirements.txt
python logger.py --port COM3 --output data/data.csv  # Collect sensor data
python train_snn.py                                   # Train SNN model
python export_weights.py                              # Export weights → firmware/include/snn_weights.h
```

### Dashboard
```bash
cd dashboard
pip install -r requirements.txt
python main.py
```

## Architecture

### Data Flow
```
ESP32 (MPU6050, 200 Hz, I2C SDA=21/SCL=22)
  → Serial 115200 baud → "timestamp_ms,accel_z\n"
    → training/logger.py → data/*.csv
      → train_snn.py → model.pth
        → export_weights.py → firmware/include/snn_weights.h
```

### Firmware Hardware Pins
- I2C: SDA=21, SCL=22
- Status LEDs: R=12, G=13, B=14 (Red=I2C failure, Green=ready)
- Sensor config: ±4G range, 94 Hz LPF, 5 ms sample interval

### Serial Data Format
CSV text, no header: `<timestamp_ms>,<accel_z_m_per_s2>\n`

The logger applies a software 94 Hz low-pass filter before writing to CSV.

## Key Dependencies

| Component | Key Libraries |
|-----------|---------------|
| Firmware  | Adafruit MPU6050 ^2.2.6, Adafruit Unified Sensor ^1.1.14 |
| Training  | torch, snntorch, numpy, pandas, scikit-learn, pyserial, jupyter |
| Dashboard | PyQt5, pyqtgraph, numpy, pyserial |

## Notes

- `.gitignore` excludes `data/*.csv`, `*.pth`, and `.pio/` build artifacts — never commit raw sensor data or trained weights
- The SNN weight export pipeline produces a C header (`snn_weights.h`) consumed directly by the firmware inference code
- `training/notebooks/` is the intended space for exploratory analysis; keep production logic in `.py` scripts
