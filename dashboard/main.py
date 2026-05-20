"""
Real-time SNN Predictive Maintenance Dashboard
================================================
Connects to the ESP32 running in INFERENCE_MODE and visualises:
  • Left  : live accel_z waveform
  • Right : output neuron membrane potential (anomaly neuron)
  • Centre: status indicator + running statistics

Serial line protocol (from firmware):
  <ts_ms>,<accel_z>                               ← raw sample  (every 5 ms)
  INF,<hidden_rate>,<anomaly_spk>,<status>,<mem2> ← inference result (every window)
"""

import sys
import collections
from typing import Optional

import numpy as np
import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

# ── Plot history length ────────────────────────────────────────────────────────
WAVEFORM_LEN = 400   # samples shown in the waveform plot
MEM_POT_LEN  = 200   # inference windows shown in the membrane-potential plot


# ═══════════════════════════════════════════════════════════════════════════════
#  Serial reader thread
# ═══════════════════════════════════════════════════════════════════════════════
class SerialReader(QtCore.QThread):
    """Reads lines from the serial port and emits typed signals."""

    raw_sample = QtCore.pyqtSignal(float, float)   # (timestamp_ms, accel_z)
    inference  = QtCore.pyqtSignal(float, float, int, float)  # (hidden_rate, anomaly_spk, status, mem2)
    error      = QtCore.pyqtSignal(str)

    def __init__(self, port: str, baud: int = 115200, parent=None):
        super().__init__(parent)
        self._port   = port
        self._baud   = baud
        self._active = True

    def stop(self):
        self._active = False

    def run(self):
        try:
            ser = serial.Serial(self._port, self._baud, timeout=1)
        except serial.SerialException as exc:
            self.error.emit(str(exc))
            return

        while self._active:
            try:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
            except serial.SerialException as exc:
                self.error.emit(str(exc))
                break

            if line.startswith("INF,"):
                parts = line[4:].split(",")
                if len(parts) == 4:
                    try:
                        self.inference.emit(
                            float(parts[0]),   # hidden_rate
                            float(parts[1]),   # anomaly_spk
                            int(parts[2]),     # status  0/1
                            float(parts[3]),   # mem2 anomaly
                        )
                    except ValueError:
                        pass
            else:
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        self.raw_sample.emit(float(parts[0]), float(parts[1]))
                    except ValueError:
                        pass

        ser.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QtWidgets.QMainWindow):

    _STATUS_STYLE = {
        "normal":  "background:#2ecc71; color:#fff; border-radius:6px; padding:8px 20px;",
        "anomaly": "background:#e74c3c; color:#fff; border-radius:6px; padding:8px 20px;",
        "offline": "background:#7f8c8d; color:#fff; border-radius:6px; padding:8px 20px;",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNN Predictive Maintenance Dashboard")
        self.resize(1200, 680)

        self._reader: Optional[SerialReader] = None

        # Circular buffers
        self._accel_buf    = collections.deque([0.0] * WAVEFORM_LEN, maxlen=WAVEFORM_LEN)
        self._mem_buf      = collections.deque([0.0] * MEM_POT_LEN,  maxlen=MEM_POT_LEN)
        self._anomaly_spks = 0
        self._normal_wins  = 0
        self._total_wins   = 0

        self._build_ui()

        # 60 Hz plot refresh
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._refresh_plots)
        self._timer.start(16)

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        pg.setConfigOptions(antialias=True, background="#1e1e2e", foreground="#cdd6f4")

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # ── Top toolbar ───────────────────────────────────────────────────────
        toolbar = QtWidgets.QHBoxLayout()

        self._port_combo = QtWidgets.QComboBox()
        self._port_combo.setMinimumWidth(120)
        self._refresh_ports()

        refresh_btn = QtWidgets.QPushButton("↻")
        refresh_btn.setFixedWidth(30)
        refresh_btn.clicked.connect(self._refresh_ports)

        self._connect_btn = QtWidgets.QPushButton("Connect")
        self._connect_btn.setCheckable(True)
        self._connect_btn.clicked.connect(self._toggle_connection)

        self._status_lbl = QtWidgets.QLabel("OFFLINE")
        self._status_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._status_lbl.setFixedHeight(36)
        self._status_lbl.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        self._status_lbl.setStyleSheet(self._STATUS_STYLE["offline"])
        self._status_lbl.setMinimumWidth(140)

        toolbar.addWidget(QtWidgets.QLabel("Port:"))
        toolbar.addWidget(self._port_combo)
        toolbar.addWidget(refresh_btn)
        toolbar.addWidget(self._connect_btn)
        toolbar.addStretch()
        toolbar.addWidget(self._status_lbl)
        outer.addLayout(toolbar)

        # ── Plot row ──────────────────────────────────────────────────────────
        plots_row = QtWidgets.QHBoxLayout()
        plots_row.setSpacing(10)

        # Waveform plot
        self._waveform_pw = pg.PlotWidget(title="Acceleration Z  (m/s²)")
        self._waveform_pw.showGrid(x=True, y=True, alpha=0.3)
        self._waveform_pw.setLabel("left", "accel_z", units="m/s²")
        self._waveform_pw.setLabel("bottom", "samples (newest →)")
        self._waveform_curve = self._waveform_pw.plot(
            pen=pg.mkPen("#89b4fa", width=1.5)
        )

        # Membrane potential plot
        self._mem_pw = pg.PlotWidget(title="Output Neuron Membrane Potential  (anomaly)")
        self._mem_pw.showGrid(x=True, y=True, alpha=0.3)
        self._mem_pw.setLabel("left", "mem potential")
        self._mem_pw.setLabel("bottom", "inference windows (newest →)")
        self._mem_curve = self._mem_pw.plot(
            pen=pg.mkPen("#f38ba8", width=1.5)
        )
        # Threshold reference line
        self._thresh_line = pg.InfiniteLine(
            pos=1.0, angle=0,
            pen=pg.mkPen("#fab387", width=1, style=QtCore.Qt.DashLine)
        )
        self._mem_pw.addItem(self._thresh_line)

        plots_row.addWidget(self._waveform_pw, stretch=3)
        plots_row.addWidget(self._mem_pw,      stretch=2)
        outer.addLayout(plots_row, stretch=1)

        # ── Stats row ─────────────────────────────────────────────────────────
        stats_row = QtWidgets.QHBoxLayout()
        stats_row.setSpacing(20)

        def _stat_widget(label: str):
            box = QtWidgets.QGroupBox(label)
            lay = QtWidgets.QVBoxLayout(box)
            lbl = QtWidgets.QLabel("—")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
            lay.addWidget(lbl)
            return box, lbl

        box1, self._lbl_windows   = _stat_widget("Windows processed")
        box2, self._lbl_anomalies = _stat_widget("Anomaly windows")
        box3, self._lbl_rate      = _stat_widget("Anomaly rate")
        box4, self._lbl_hidden    = _stat_widget("Hidden spike rate")

        for b in (box1, box2, box3, box4):
            stats_row.addWidget(b)
        outer.addLayout(stats_row)

    # ── Port management ────────────────────────────────────────────────────────
    def _refresh_ports(self):
        self._port_combo.clear()
        for p in serial.tools.list_ports.comports():
            self._port_combo.addItem(p.device)

    def _toggle_connection(self, checked: bool):
        if checked:
            port = self._port_combo.currentText()
            if not port:
                self._connect_btn.setChecked(False)
                return
            self._reader = SerialReader(port)
            self._reader.raw_sample.connect(self._on_raw_sample)
            self._reader.inference.connect(self._on_inference)
            self._reader.error.connect(self._on_serial_error)
            self._reader.start()
            self._connect_btn.setText("Disconnect")
            self._set_status("normal")
        else:
            self._disconnect()

    def _disconnect(self):
        if self._reader:
            self._reader.stop()
            self._reader.wait(2000)
            self._reader = None
        self._connect_btn.setText("Connect")
        self._connect_btn.setChecked(False)
        self._set_status("offline")

    # ── Data slots ────────────────────────────────────────────────────────────
    def _on_raw_sample(self, _ts: float, az: float):
        self._accel_buf.append(az)

    def _on_inference(self, hidden_rate: float, anomaly_spk: float,
                      status: int, mem2: float):
        self._mem_buf.append(mem2)
        self._total_wins += 1
        if status == 1:
            self._anomaly_spks += 1
        else:
            self._normal_wins += 1

        self._set_status("anomaly" if status == 1 else "normal")

        anomaly_rate = self._anomaly_spks / self._total_wins if self._total_wins else 0.0
        self._lbl_windows.setText(str(self._total_wins))
        self._lbl_anomalies.setText(str(self._anomaly_spks))
        self._lbl_rate.setText(f"{anomaly_rate*100:.1f} %")
        self._lbl_hidden.setText(f"{hidden_rate:.2f}")

    def _on_serial_error(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Serial error", msg)
        self._disconnect()

    # ── Plot refresh ──────────────────────────────────────────────────────────
    def _refresh_plots(self):
        self._waveform_curve.setData(np.array(self._accel_buf))
        self._mem_curve.setData(np.array(self._mem_buf))

    # ── Status helper ─────────────────────────────────────────────────────────
    def _set_status(self, state: str):
        text = {"normal": "NORMAL", "anomaly": "⚠ ANOMALY", "offline": "OFFLINE"}
        self._status_lbl.setText(text[state])
        self._status_lbl.setStyleSheet(self._STATUS_STYLE[state])

    def closeEvent(self, event):
        self._disconnect()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window,      QtGui.QColor(30, 30, 46))
    palette.setColor(QtGui.QPalette.WindowText,  QtGui.QColor(205, 214, 244))
    palette.setColor(QtGui.QPalette.Base,        QtGui.QColor(24, 24, 37))
    palette.setColor(QtGui.QPalette.Button,      QtGui.QColor(49, 50, 68))
    palette.setColor(QtGui.QPalette.ButtonText,  QtGui.QColor(205, 214, 244))
    palette.setColor(QtGui.QPalette.Highlight,   QtGui.QColor(137, 180, 250))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
