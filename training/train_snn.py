"""
SNN training for vibration-based predictive maintenance.

Pipeline
--------
  CSV (normal.csv / anomaly.csv)
    → delta modulation  →  binary spike windows  [batch, WINDOW]
    → rate-coded BPTT   →  VibrationSNN
    → model.pth

Network:  Linear(WINDOW→HIDDEN) + Leaky  →  Linear(HIDDEN→2) + Leaky
Binary input spikes allow MAC-free accumulation on the ESP32 edge device.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import snntorch as snn
from snntorch import surrogate

# ── Defaults ──────────────────────────────────────────────────────────────────
WINDOW       = 128     # samples per window  (0.64 s @ 200 Hz)
STRIDE       = 64      # 50 % overlap → ~2× data density
DELTA_THRESH = 0.10    # m/s²  — tune after inspecting your CSV
BETA         = 0.9     # initial LIF membrane leak  (learned during training)
THRESHOLD    = 1.0     # LIF firing threshold
T_STEPS      = 16      # timesteps to present each window (rate-coded repetition)
HIDDEN       = 32
BATCH        = 32
EPOCHS       = 60
LR           = 1e-3

DATA_DIR     = pathlib.Path("data")
MODEL_PATH   = pathlib.Path("model.pth")


# ── Delta modulation ──────────────────────────────────────────────────────────

def delta_encode(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a continuous 1-D signal to a binary spike train.

    A spike (1) is emitted whenever the absolute difference between
    consecutive samples exceeds *threshold*.  The first sample is always 0.
    """
    spikes = np.zeros(len(signal), dtype=np.float32)
    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i - 1]) > threshold:
            spikes[i] = 1.0
    return spikes


def make_windows(spikes: np.ndarray, label: int, window: int, stride: int):
    X, y = [], []
    for start in range(0, len(spikes) - window + 1, stride):
        X.append(spikes[start : start + window])
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(data_dir: pathlib.Path, delta_thresh: float, window: int, stride: int):
    # idle.csv is optional; treated as normal (label 0) when present
    required = {"normal.csv": 0, "anomaly.csv": 1}
    optional = {"idle.csv":   0}
    Xs, ys = [], []
    for fname, label in {**required, **optional}.items():
        path = data_dir / fname
        if not path.exists():
            if fname in optional:
                continue
            raise FileNotFoundError(
                f"Missing training file: {path}\n"
                "Collect data first:  python logger.py --port COM5 --output data/normal.csv"
            )
        df     = pd.read_csv(path)
        signal = df["accel_z"].to_numpy(dtype=np.float32)
        spikes = delta_encode(signal, delta_thresh)
        X, y   = make_windows(spikes, label, window, stride)
        Xs.append(X)
        ys.append(y)
        spike_rate = spikes.mean()
        print(f"  {fname:15s}: {len(signal):6d} raw samples  →  {len(X):4d} windows"
              f"  (spike rate {spike_rate:.3f})")
    return np.concatenate(Xs), np.concatenate(ys)


# ── SNN model ─────────────────────────────────────────────────────────────────

class VibrationSNN(nn.Module):
    """Two-layer fully-connected SNN with learned LIF leak rates."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 beta: float, threshold: float):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1  = nn.Linear(n_in,     n_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, learn_beta=True)
        self.fc2  = nn.Linear(n_hidden, n_out,    bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, learn_beta=True)

    def forward(self, x: torch.Tensor, t_steps: int) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : [batch, n_in]  binary spike pattern (float 0/1)
        t_steps : number of times the same pattern is presented (rate coding)

        Returns
        -------
        spk_count : [batch, n_out]  output spike counts over t_steps
        """
        mem1      = self.lif1.init_leaky()
        mem2      = self.lif2.init_leaky()
        spk_count = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)

        for _ in range(t_steps):
            cur1       = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2       = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_count  = spk_count + spk2

        return spk_count   # argmax → predicted class; higher count = more confident


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, t_steps):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(xb, t_steps)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device, t_steps):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        out = model(xb.to(device), t_steps)
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(yb.tolist())
    return preds, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SNN for vibration anomaly detection")
    parser.add_argument("--data-dir",     default=str(DATA_DIR),  help="Directory with normal.csv / anomaly.csv")
    parser.add_argument("--model-out",    default=str(MODEL_PATH), help="Output model path")
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--batch",        type=int,   default=BATCH)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--delta-thresh", type=float, default=DELTA_THRESH,
                        help="Delta modulation threshold in m/s²")
    parser.add_argument("--hidden",       type=int,   default=HIDDEN)
    parser.add_argument("--t-steps",      type=int,   default=T_STEPS)
    parser.add_argument("--window",       type=int,   default=WINDOW)
    parser.add_argument("--stride",       type=int,   default=STRIDE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading dataset ...")
    X, y = load_dataset(
        pathlib.Path(args.data_dir), args.delta_thresh, args.window, args.stride
    )
    print(f"  Total windows : {len(X)}  (normal={int((y==0).sum())}, anomaly={int((y==1).sum())})")
    print(f"  Overall spike density : {X.mean():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    tr_ds = torch.utils.data.TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    te_ds = torch.utils.data.TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = VibrationSNN(args.window, args.hidden, 2, BETA, THRESHOLD).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters : {n_params}  ({n_params*4/1024:.1f} KB as float32)")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, tr_loader, optimizer, criterion, device, args.t_steps)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{epoch:3d}/{args.epochs}]  loss={loss:.4f}  train_acc={acc:.3f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    preds, labels = evaluate(model, te_loader, device, args.t_steps)
    print("\nTest-set classification report:")
    print(classification_report(labels, preds, target_names=["normal", "anomaly"], digits=4))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = pathlib.Path(args.model_out)
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "window":       args.window,
            "hidden":       args.hidden,
            "t_steps":      args.t_steps,
            "beta":         BETA,
            "threshold":    THRESHOLD,
            "delta_thresh": args.delta_thresh,
        },
    }, out_path)
    print(f"Model saved → {out_path}")


if __name__ == "__main__":
    main()
