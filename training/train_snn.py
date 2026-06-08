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
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

import snntorch as snn
from snntorch import surrogate

# ── Defaults ──────────────────────────────────────────────────────────────────
WINDOW       = 128     # samples per window  (0.64 s @ 200 Hz)
STRIDE       = 64      # 50 % overlap → ~2× data density
DELTA_THRESH = 0.10    # m/s²  — tune after inspecting your CSV
BETA         = 0.9     # initial LIF membrane leak  (learned during training)
THRESHOLD    = 0.1     # LIF firing threshold
T_STEPS      = 4       # timesteps to present each window (rate-coded repetition)
HIDDEN       = 32
BATCH        = 32
EPOCHS             = 100
LR                 = 1e-3
CHUNK_SIZE         = 256   # timesteps per training chunk  (streaming mode)
DECISION_INTERVAL  = 200   # samples between classifications on ESP32 (≈ 1 s @ 200 Hz)

DATA_DIR     = pathlib.Path("data")
MODEL_PATH   = pathlib.Path("model.pth")

CWRU_HZ    = 12000     # DE_time channel sample rate
CWRU_G2MS2 = 9.80665   # CWRU data is in g → convert to m/s² for consistent delta_thresh
ESP32_HZ   = 200       # ESP32 / MPU6050 sample rate


# ── CWRU resampling ───────────────────────────────────────────────────────────

def _cwru_resample(signal: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """Resample CWRU signal from src_hz to dst_hz and convert g → m/s²."""
    from math import gcd
    from scipy.signal import resample_poly
    scaled = signal * CWRU_G2MS2
    if src_hz == dst_hz:
        return scaled.astype(np.float32)
    g    = gcd(src_hz, dst_hz)
    up   = dst_hz // g
    down = src_hz // g
    return resample_poly(scaled, up, down).astype(np.float32)


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

def load_dataset(data_dir: pathlib.Path, delta_thresh: float, window: int, stride: int,
                 fault_file: str = "idle.csv", normal_file: str = "normal.csv"):
    label_map = {fault_file: 0, normal_file: 1}
    Xs, ys = [], []
    for fname, label in label_map.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing training file: {path}\n"
                f"Collect data first:  python dump_flash.py --port COM5 --output data/{fname}"
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


# ── CWRU dataset loader ───────────────────────────────────────────────────────

def load_cwru_dataset(normal_path: pathlib.Path, fault_path: pathlib.Path,
                      delta_thresh: float, window: int, stride: int,
                      src_hz: int = CWRU_HZ, target_hz: int = ESP32_HZ):
    """Load CWRU bearing .mat files and return (X, y) spike windows.

    Data is resampled from src_hz (default 12 kHz) to target_hz (default 200 Hz)
    and converted from g to m/s² before delta encoding, so delta_thresh values
    are directly comparable to ESP32 / MPU6050 data.

    Download: https://engineering.case.edu/bearingdatacenter/download-data-file
    Suggested files:
      Normal → 97.mat   (0 HP baseline)
      Fault  → 105.mat  (0.007-inch inner-race fault, 0 HP)
               118.mat  (0.007-inch ball fault)
               130.mat  (0.007-inch outer-race fault)
    """
    import scipy.io as sio

    def _load_mat(path: pathlib.Path) -> np.ndarray:
        mat  = sio.loadmat(str(path))
        keys = [k for k in mat if "DE_time" in k]   # prefer drive-end channel
        key  = keys[0] if keys else next(k for k in mat if not k.startswith("_"))
        return mat[key].flatten().astype(np.float32)

    Xs, ys = [], []
    for path, label, name in [(normal_path, 1, "normal"), (fault_path, 0, "fault ")]:
        if not path.exists():
            raise FileNotFoundError(
                f"CWRU file not found: {path}\n"
                "Download from https://engineering.case.edu/bearingdatacenter/download-data-file"
            )
        raw    = _load_mat(path)
        signal = _cwru_resample(raw, src_hz, target_hz)
        spikes = delta_encode(signal, delta_thresh)
        X, y   = make_windows(spikes, label, window, stride)
        Xs.append(X); ys.append(y)
        print(f"  {name}: {len(raw):8d} raw @ {src_hz} Hz  →  {len(signal):6d} @ {target_hz} Hz"
              f"  →  {len(X):4d} windows  (spike rate {spikes.mean():.3f})")

    return np.concatenate(Xs), np.concatenate(ys)


# ── SNN model ─────────────────────────────────────────────────────────────────

class VibrationSNN(nn.Module):
    """LIF hidden layer + linear output.

    Removing LIF2 eliminates output binarisation so the CE loss always has
    a non-zero gradient regardless of spike density.  The firmware mirrors
    this by accumulating fc2(spk1) directly instead of an integrate-and-fire
    loop at the output.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 beta: float, threshold: float):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=5)
        self.fc1  = nn.Linear(n_in,     n_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, learn_beta=False)
        self.fc2  = nn.Linear(n_hidden, n_out,    bias=False)

    def forward(self, x: torch.Tensor, t_steps: int) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : [batch, n_in]  binary spike pattern (float 0/1)
        t_steps : number of times the same pattern is presented

        Returns
        -------
        out_sum : [batch, n_out]  accumulated fc2 outputs — used as logits
        """
        mem1    = self.lif1.init_leaky()
        out_sum = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)

        for _ in range(t_steps):
            cur1       = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            out_sum    = out_sum + self.fc2(spk1)

        return out_sum   # argmax → predicted class


# ── Streaming SNN ────────────────────────────────────────────────────────────

class StreamingVibrationSNN(nn.Module):
    """Truly streaming SNN: one binary spike per timestep, no fixed window.

    At inference time mem1 is never reset — the LIF continuously integrates
    the incoming spike train.  FC1 has only one input (the current spike),
    so all temporal pattern discrimination comes from the LIF dynamics.
    learn_beta=True lets the single shared beta optimise during training.
    """

    def __init__(self, n_hidden: int, n_out: int, beta: float, threshold: float):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=5)
        self.fc1  = nn.Linear(1, n_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, learn_beta=True)
        self.fc2  = nn.Linear(n_hidden, n_out, bias=False)

    def forward(self, spike_seq: torch.Tensor, mem1: torch.Tensor):
        """
        Parameters
        ----------
        spike_seq : [T, batch, 1]  one binary value per timestep
        mem1      : [batch, n_hidden]  initial membrane potential

        Returns
        -------
        out_sum : [batch, n_out]  accumulated fc2 logits
        mem1    : [batch, n_hidden]  final membrane potential
        """
        out_sum = torch.zeros(spike_seq.size(1), self.fc2.out_features,
                              device=spike_seq.device)
        for t in range(spike_seq.size(0)):
            cur1       = self.fc1(spike_seq[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            out_sum    = out_sum + self.fc2(spk1)
        return out_sum, mem1


# ── Streaming dataset helpers ─────────────────────────────────────────────────

def _signal_to_chunks(signal: np.ndarray, label: int,
                      delta_thresh: float, chunk_size: int):
    spikes = delta_encode(signal, delta_thresh)
    chunks = [
        (spikes[s:s + chunk_size].copy(), label)
        for s in range(0, len(spikes) - chunk_size + 1, chunk_size)
    ]
    return chunks, float(spikes.mean())


def load_stream_dataset_csv(data_dir: pathlib.Path,
                             delta_thresh: float, chunk_size: int,
                             fault_file: str = "idle.csv", normal_file: str = "normal.csv"):
    label_map = {fault_file: 0, normal_file: 1}
    all_chunks: list = []
    for fname, label in label_map.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        signal = pd.read_csv(path)["accel_z"].to_numpy(dtype=np.float32)
        chunks, rate = _signal_to_chunks(signal, label, delta_thresh, chunk_size)
        all_chunks.extend(chunks)
        print(f"  {fname:15s}: {len(signal):6d} samples → {len(chunks):4d} chunks"
              f"  (spike rate {rate:.3f})")
    return all_chunks


def load_stream_dataset_cwru(normal_path: pathlib.Path, fault_path: pathlib.Path,
                              delta_thresh: float, chunk_size: int,
                              src_hz: int = CWRU_HZ, target_hz: int = ESP32_HZ):
    import scipy.io as sio

    def _load_mat(path: pathlib.Path) -> np.ndarray:
        mat  = sio.loadmat(str(path))
        keys = [k for k in mat if "DE_time" in k]
        key  = keys[0] if keys else next(k for k in mat if not k.startswith("_"))
        return mat[key].flatten().astype(np.float32)

    all_chunks: list = []
    for path, label, name in [(normal_path, 1, "normal"), (fault_path, 0, "fault ")]:
        if not path.exists():
            raise FileNotFoundError(
                f"CWRU file not found: {path}\n"
                "Download:  python download_cwru.py"
            )
        raw    = _load_mat(path)
        signal = _cwru_resample(raw, src_hz, target_hz)
        chunks, rate = _signal_to_chunks(signal, label, delta_thresh, chunk_size)
        all_chunks.extend(chunks)
        print(f"  {name}: {len(raw):8d} raw @ {src_hz} Hz  →  {len(signal):6d} @ {target_hz} Hz"
              f"  →  {len(chunks):4d} chunks  (spike rate {rate:.3f})")
    return all_chunks


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


def train_epoch_stream(model, chunks, optimizer, criterion, device, hidden):
    model.train()
    random.shuffle(chunks)
    total_loss, correct, n = 0.0, 0, 0
    for spike_chunk, label in chunks:
        x    = torch.tensor(spike_chunk, dtype=torch.float32).view(-1, 1, 1).to(device)
        y    = torch.tensor([label], device=device)
        mem1 = torch.zeros(1, hidden, device=device)
        optimizer.zero_grad()
        out, _ = model(x, mem1)
        loss   = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
        n += 1
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_stream(model, chunks, device, hidden, return_scores=False):
    model.eval()
    preds, labels_out, scores = [], [], []
    for spike_chunk, label in chunks:
        x    = torch.tensor(spike_chunk, dtype=torch.float32).view(-1, 1, 1).to(device)
        mem1 = torch.zeros(1, hidden, device=device)
        out, _ = model(x, mem1)
        preds.append(out.argmax(1).item())
        labels_out.append(label)
        if return_scores:
            scores.append(torch.softmax(out, dim=1)[0, 1].item())  # P(normal=1)
    if return_scores:
        return preds, labels_out, scores
    return preds, labels_out


# ── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate_stream(all_chunks, args, device, class_names):
    """Stratified K-fold cross-validation for the streaming SNN."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  roc_auc_score)

    chunk_data = [c for c, _ in all_chunks]
    all_labels = [y for _, y in all_chunks]
    k          = args.cv

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics   = []
    agg_preds, agg_true, agg_scores = [], [], []
    loss_histories = []

    for fi, (tr_idx, va_idx) in enumerate(skf.split(chunk_data, all_labels)):
        train_ch = [(chunk_data[i], all_labels[i]) for i in tr_idx]
        val_ch   = [(chunk_data[i], all_labels[i]) for i in va_idx]
        n0 = sum(1 for _, y in train_ch if y == 0)
        n1 = len(train_ch) - n0

        model_f   = StreamingVibrationSNN(args.hidden, 2, BETA, THRESHOLD).to(device)
        opt_f     = torch.optim.Adam(model_f.parameters(), lr=args.lr)
        wf        = torch.tensor([1.0/n0, 1.0/n1], dtype=torch.float32, device=device)
        wf        = wf / wf.sum() * 2
        crit_f    = nn.CrossEntropyLoss(weight=wf)
        sched_f   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=args.epochs)

        best_vacc, patience_left, best_st = -1.0, args.patience, None
        fold_losses = []
        for epoch in range(1, args.epochs + 1):
            loss, _ = train_epoch_stream(model_f, train_ch, opt_f, crit_f, device, args.hidden)
            sched_f.step()
            fold_losses.append(loss)
            vp, vl = evaluate_stream(model_f, val_ch, device, args.hidden)
            vacc = sum(p == l for p, l in zip(vp, vl)) / len(vl)
            if vacc > best_vacc:
                best_vacc, patience_left = vacc, args.patience
                best_st = {k_: v.clone() for k_, v in model_f.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left == 0:
                    model_f.load_state_dict(best_st)
                    break

        loss_histories.append(fold_losses)
        vp, vl, vs = evaluate_stream(model_f, val_ch, device, args.hidden, return_scores=True)
        agg_preds.extend(vp); agg_true.extend(vl); agg_scores.extend(vs)

        acc  = accuracy_score(vl, vp)
        prec, rec, f1, _ = precision_recall_fscore_support(vl, vp, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(vl, vs)
        except ValueError:
            auc = float('nan')
        fold_metrics.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc})
        print(f"  Fold {fi+1}/{k}:  acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  "
              f"F1={f1:.4f}  AUC={auc:.4f}  epochs={len(fold_losses)}")

    print(f"\n{k}-Fold CV Summary (mean ± std):")
    for key, name in [('acc','Accuracy'),('prec','Precision'),
                      ('rec','Recall'),('f1','F1 Score'),('auc','ROC-AUC')]:
        vals = [m[key] for m in fold_metrics if m[key] == m[key]]
        print(f"  {name:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return fold_metrics, agg_preds, agg_true, agg_scores, loss_histories


def save_plots(loss_histories, all_true, all_preds, all_scores, class_names, plot_dir):
    """Save convergence plot, confusion matrix, and ROC curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    plot_dir = pathlib.Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convergence curves
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, losses in enumerate(loss_histories):
        ax.plot(losses, label=f'Fold {i+1}', alpha=0.75, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title(f'Training Convergence ({len(loss_histories)}-Fold CV)', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(plot_dir / 'convergence.png', dpi=150); plt.close()

    # 2. Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set_xticks(list(ticks)); ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(list(ticks)); ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Aggregated CV)', fontsize=12)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                    fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout(); fig.savefig(plot_dir / 'confusion_matrix.png', dpi=150); plt.close()

    # 3. ROC curve
    try:
        fpr, tpr, _ = roc_curve(all_true, all_scores)
        roc_auc = sk_auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0,1],[0,1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve (Aggregated K-Fold CV)', fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout(); fig.savefig(plot_dir / 'roc_curve.png', dpi=150); plt.close()
    except Exception:
        pass

    print(f"Plots saved to {plot_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SNN for vibration anomaly detection")
    parser.add_argument("--mode",         choices=["windowed", "streaming"], default="windowed",
                        help="'windowed': fixed-size input (original); 'streaming': one spike/step")
    parser.add_argument("--chunk-size",        type=int, default=CHUNK_SIZE,
                        help="[streaming] timesteps per training chunk")
    parser.add_argument("--decision-interval", type=int, default=DECISION_INTERVAL,
                        help="[streaming] samples between ESP32 classifications")
    parser.add_argument("--patience",          type=int, default=10,
                        help="[streaming] early-stop after this many epochs without val improvement")
    parser.add_argument("--dataset",      choices=["csv", "cwru"], default="csv",
                        help="Data source: 'csv' (custom ESP32 data) or 'cwru' (.mat files)")
    parser.add_argument("--data-dir",     default=str(DATA_DIR),  help="[csv] Directory with CSV data files")
    parser.add_argument("--fault-file",   default="anomaly.csv",  help="[csv] Filename for fault/anomaly class (label 0)")
    parser.add_argument("--normal-file",  default="normal.csv",   help="[csv] Filename for normal class (label 1)")
    parser.add_argument("--cwru-normal",    default="data/cwru/97.mat",
                        help="[cwru] Path to normal baseline .mat file (e.g. 97.mat)")
    parser.add_argument("--cwru-fault",     default="data/cwru/105.mat",
                        help="[cwru] Path to fault .mat file (e.g. 105.mat for inner-race fault)")
    parser.add_argument("--cwru-source-hz", type=int, default=CWRU_HZ,
                        help="[cwru] Sample rate of the .mat files (default 12000 Hz)")
    parser.add_argument("--target-hz",      type=int, default=ESP32_HZ,
                        help="[cwru] Resample target Hz — must match ESP32 rate (default 200 Hz)")
    parser.add_argument("--model-out",    default=str(MODEL_PATH), help="Output model path")
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--batch",        type=int,   default=BATCH)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--delta-thresh", type=float, default=DELTA_THRESH,
                        help="Delta modulation threshold (m/s² for csv; tune for cwru units)")
    parser.add_argument("--hidden",       type=int,   default=HIDDEN)
    parser.add_argument("--t-steps",      type=int,   default=T_STEPS)
    parser.add_argument("--window",       type=int,   default=WINDOW)
    parser.add_argument("--stride",       type=int,   default=STRIDE)
    parser.add_argument("--cv",           type=int,   default=5,
                        help="[streaming] K-fold cross-validation (0 = disable, use 80/20 split)")
    parser.add_argument("--plot-dir",     default="plots",
                        help="Directory to save convergence/confusion/ROC plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STREAMING MODE
    # ══════════════════════════════════════════════════════════════════════════
    if args.mode == "streaming":
        if args.dataset == "cwru":
            class_names = ["fault", "normal"]
            print("\nLoading CWRU streaming dataset ...")
            all_chunks = load_stream_dataset_cwru(
                pathlib.Path(args.cwru_normal), pathlib.Path(args.cwru_fault),
                args.delta_thresh, args.chunk_size,
                src_hz=args.cwru_source_hz, target_hz=args.target_hz,
            )
        else:
            class_names = [args.fault_file.replace(".csv",""), "normal"]
            print("\nLoading streaming dataset ...")
            all_chunks = load_stream_dataset_csv(
                pathlib.Path(args.data_dir), args.delta_thresh, args.chunk_size,
                fault_file=args.fault_file, normal_file=args.normal_file,
            )

        n0 = sum(1 for _, y in all_chunks if y == 0)
        n1 = len(all_chunks) - n0
        print(f"  Total chunks : {len(all_chunks)}  ({class_names[0]}={n0}, normal={n1})")
        for lbl, name in [(0, class_names[0]), (1, "normal")]:
            sc = np.array([c.sum() for c, y in all_chunks if y == lbl])
            if len(sc):
                print(f"  {name:12s} spike counts/chunk: "
                      f"min={sc.min():.0f}  mean={sc.mean():.1f}  max={sc.max():.0f}  "
                      f"zero-spike={int((sc == 0).sum())}/{len(sc)}")

        n_params = sum(p.numel() for p in
                       StreamingVibrationSNN(args.hidden, 2, BETA, THRESHOLD).parameters())
        print(f"\nModel parameters : {n_params}  ({n_params*4/1024:.2f} KB as float32)")

        # ── Cross-validation ──────────────────────────────────────────────────
        if args.cv > 1:
            print(f"\n{'='*60}")
            print(f"  {args.cv}-Fold Stratified Cross-Validation")
            print(f"{'='*60}")
            fold_metrics, cv_preds, cv_true, cv_scores, loss_histories = \
                cross_validate_stream(all_chunks, args, device, class_names)
            print("\nAggregated CV Classification Report:")
            print(classification_report(cv_true, cv_preds, target_names=class_names, digits=4))
            if args.plot_dir:
                save_plots(loss_histories, cv_true, cv_preds, cv_scores,
                           class_names, args.plot_dir)

        # ── Final model (full data if CV ran, else 80/20 split) ───────────────
        random.seed(42)
        random.shuffle(all_chunks)
        if args.cv > 1:
            train_chunks = all_chunks          # full dataset → deployment model
            test_chunks  = []
            print(f"\nFinal model: training on all {len(train_chunks)} chunks ...")
        else:
            split        = int(len(all_chunks) * 0.8)
            train_chunks = all_chunks[:split]
            test_chunks  = all_chunks[split:]
            print(f"\nTraining on {len(train_chunks)} chunks (80/20 split) ...")
        n0_tr = sum(1 for _, y in train_chunks if y == 0)
        n1_tr = len(train_chunks) - n0_tr

        model     = StreamingVibrationSNN(args.hidden, 2, BETA, THRESHOLD).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        w  = torch.tensor([1.0 / n0_tr, 1.0 / n1_tr], dtype=torch.float32, device=device)
        w  = w / w.sum() * 2
        criterion = nn.CrossEntropyLoss(weight=w)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        has_val = len(test_chunks) > 0
        if has_val:
            print(f"Early stop patience={args.patience} ...")
        best_val_acc  = -1.0
        patience_left = args.patience
        for epoch in range(1, args.epochs + 1):
            loss, acc = train_epoch_stream(model, train_chunks, optimizer, criterion,
                                           device, args.hidden)
            scheduler.step()
            if has_val:
                val_preds, val_labels = evaluate_stream(model, test_chunks, device, args.hidden)
                val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)
                if epoch % 5 == 0 or epoch == 1:
                    print(f"  [{epoch:3d}/{args.epochs}]  loss={loss:.4f}  "
                          f"train_acc={acc:.3f}  val_acc={val_acc:.3f}")
                if val_acc > best_val_acc:
                    best_val_acc  = val_acc
                    patience_left = args.patience
                    best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        print(f"  Early stop at epoch {epoch}  (best val_acc={best_val_acc:.4f})")
                        model.load_state_dict(best_state)
                        break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    print(f"  [{epoch:3d}/{args.epochs}]  loss={loss:.4f}  train_acc={acc:.3f}")

        tr_preds, tr_labels = evaluate_stream(model, train_chunks, device, args.hidden)
        print("\nTrain-set classification report:")
        print(classification_report(tr_labels, tr_preds, target_names=class_names, digits=4))

        if test_chunks:
            te_preds, te_labels = evaluate_stream(model, test_chunks, device, args.hidden)
            print("Test-set classification report:")
            print(classification_report(te_labels, te_preds, target_names=class_names, digits=4))

        out_path = pathlib.Path(args.model_out)
        torch.save({
            "model_state": model.state_dict(),
            "config": {
                "mode":              "streaming",
                "hidden":            args.hidden,
                "beta":              BETA,
                "threshold":         THRESHOLD,
                "delta_thresh":      args.delta_thresh,
                "decision_interval": args.decision_interval,
            },
        }, out_path)
        print(f"Model saved → {out_path}")
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  WINDOWED MODE  (original)
    # ══════════════════════════════════════════════════════════════════════════
    # ── Data ──────────────────────────────────────────────────────────────────
    if args.dataset == "cwru":
        class_names = ["fault", "normal"]
        print("\nLoading CWRU dataset ...")
        X, y = load_cwru_dataset(
            pathlib.Path(args.cwru_normal), pathlib.Path(args.cwru_fault),
            args.delta_thresh, args.window, args.stride,
            src_hz=args.cwru_source_hz, target_hz=args.target_hz,
        )
    else:
        class_names = [args.fault_file.replace(".csv",""), "normal"]
        print("\nLoading dataset ...")
        X, y = load_dataset(
            pathlib.Path(args.data_dir), args.delta_thresh, args.window, args.stride,
            fault_file=args.fault_file, normal_file=args.normal_file,
        )

    print(f"  Total windows : {len(X)}  ({class_names[0]}={int((y==0).sum())}, normal={int((y==1).sum())})")
    print(f"  Overall spike density : {X.mean():.3f}")

    # ── Spike-count diagnostics ───────────────────────────────────────────────
    for lbl, name in [(0, class_names[0]), (1, "normal")]:
        sc = X[y == lbl].sum(axis=1)
        print(f"  {name:12s} spike counts/window : "
              f"min={sc.min():.0f}  mean={sc.mean():.1f}  max={sc.max():.0f}  "
              f"zero-spike windows={int((sc == 0).sum())}/{int((y==lbl).sum())}")

    # ── Logistic regression baseline (separability check) ────────────────────
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    lr.fit(X, y)
    lr_preds = lr.predict(X)
    print("\nLogistic regression (full-data, sanity check):")
    print(classification_report(y, lr_preds, target_names=class_names, digits=4))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    n0 = int((y_tr == 0).sum())
    n1 = int((y_tr == 1).sum())

    tr_ds = torch.utils.data.TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    te_ds = torch.utils.data.TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    # Balanced sampling: each batch has ~50 % of each class regardless of dataset imbalance
    sample_w = np.where(y_tr == 0, 1.0 / n0, 1.0 / n1).tolist()
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(y_tr), replacement=True)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, sampler=sampler, drop_last=True)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = VibrationSNN(args.window, args.hidden, 2, BETA, THRESHOLD).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Inverse-frequency weights: upweight minority class (normal running)
    w  = torch.tensor([1.0 / n0, 1.0 / n1], dtype=torch.float32, device=device)
    w  = w / w.sum() * 2          # normalise so weights sum to n_classes
    criterion = nn.CrossEntropyLoss(weight=w)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters : {n_params}  ({n_params*4/1024:.1f} KB as float32)")

    # ── Training ──────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"\nTraining for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, tr_loader, optimizer, criterion, device, args.t_steps)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{epoch:3d}/{args.epochs}]  loss={loss:.4f}  train_acc={acc:.3f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    tr_preds, tr_labels = evaluate(model, tr_loader, device, args.t_steps)
    print("\nTrain-set classification report:")
    print(classification_report(tr_labels, tr_preds, target_names=class_names, digits=4))

    preds, labels = evaluate(model, te_loader, device, args.t_steps)
    print("Test-set classification report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

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
